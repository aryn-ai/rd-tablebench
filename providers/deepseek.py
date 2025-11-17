#!/usr/bin/env python3
import base64
import io
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import modal

# Modal app
app = modal.App("ocr-benchmark-deepseek")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
BATCH_SIZE = 8
PROMPT = "<image>\nFree OCR."
GPU_TYPE = "A100-80GB"
TIMEOUT = 7200  # 2 hours
PDF_DPI = 200
MAX_RETRIES = 3

# VLLM image optimized for DeepSeek-OCR with model pre-downloading
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",  # CUDA 12.1
    )
    .pip_install(
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "einops",
        "addict",
        "easydict",
    )
    .run_commands(
        # Install vLLM from nightly build (required for DeepSeek-OCR)
        # vLLM includes flash-attn as a dependency
        "pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly",
    )
    .pip_install(
        "Pillow",
        "PyMuPDF",
        "huggingface-hub",
    )
    .run_commands(
        # Pre-download the model to avoid network issues during runtime
        f"huggingface-cli download {MODEL_NAME} --local-dir /models/huggingface/hub/deepseek-ocr",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
        }
    )
    .workdir("/root")
)


@app.cls(
    gpu=GPU_TYPE,
    timeout=TIMEOUT,
    image=vllm_image,
)
class DeepSeekOCRProcessor:
    """DeepSeek OCR Processor using vLLM with optimal configuration."""

    @modal.enter()
    def start_model(self):
        """Initialize the vLLM model."""
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        print("=" * 80)
        print("Initializing DeepSeek-OCR model...")
        print("=" * 80)

        # Create model instance
        self.llm = LLM(
            model=MODEL_NAME,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            download_dir="/models/huggingface",
            trust_remote_code=True,
        )

        # Prepare sampling parameters
        # ngram logit processor helps generate better structured HTML tables
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )

        print("✓ Model initialized successfully")
        print("=" * 80)

    def _process_single_image(self, img_data: Dict, retry_count: int = 0) -> Dict:
        """
        Process a single image.

        Args:
            img_data: Dict with 'filename' and 'image_pil' (PIL Image)
            retry_count: Current retry attempt

        Returns:
            Dict with results including table HTML, processing time, and any errors
        """
        filename = img_data["filename"]
        image_pil = img_data["image_pil"]

        try:
            # Prepare input
            model_input = {"prompt": PROMPT, "multi_modal_data": {"image": image_pil}}

            start_time = time.time()
            # Generate output
            model_outputs = self.llm.generate([model_input], self.sampling_params)
            latency = time.time() - start_time

            # Extract response content
            content = model_outputs[0].outputs[0].text

            # Debug: Print first response to see format
            if not hasattr(self, "_printed_sample"):
                print(f"\n{'=' * 60}")
                print(f"Sample response from {filename}:")
                print(f"{'=' * 60}")
                print(content[:500] if len(content) > 500 else content)
                print(f"{'=' * 60}\n")
                self._printed_sample = True

            # Extract and clean table HTML
            table_html = self._extract_table_html(content)

            # Get token counts
            prompt_tokens = (
                len(model_outputs[0].prompt_token_ids)
                if hasattr(model_outputs[0], "prompt_token_ids")
                else 0
            )
            completion_tokens = (
                len(model_outputs[0].outputs[0].token_ids)
                if hasattr(model_outputs[0].outputs[0], "token_ids")
                else 0
            )

            result = {
                "filename": filename,
                "success": True,
                "table_html": table_html,
                "raw_content": content,
                "processing_time": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "model": MODEL_NAME,
                "error": None,
            }

            return result

        except Exception as e:
            error_msg = str(e)

            # Retry logic for transient errors
            if retry_count < MAX_RETRIES and (
                "timeout" in error_msg.lower()
                or "connection" in error_msg.lower()
                or "network" in error_msg.lower()
                or "cuda" in error_msg.lower()
            ):
                wait_time = 2**retry_count  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"  ⚠ {filename}: Retry {retry_count + 1}/{MAX_RETRIES} after {wait_time}s - {error_msg}"
                )
                time.sleep(wait_time)
                return self._process_single_image(img_data, retry_count + 1)

            # Failed after retries or non-retryable error
            return {
                "filename": filename,
                "success": False,
                "table_html": "<table></table>",
                "raw_content": "",
                "processing_time": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": MODEL_NAME,
                "error": error_msg,
            }

    def _extract_table_html(self, content: str) -> str:
        """
        Extract and convert table from DeepSeek OCR response to HTML.

        DeepSeek-OCR is optimized for HTML table generation and typically
        returns well-structured HTML directly.

        Args:
            content: Raw response content from vLLM

        Returns:
            HTML table string
        """
        # Look for HTML table tags
        table_match = re.search(
            r"<table[^>]*>.*?</table>", content, re.DOTALL | re.IGNORECASE
        )

        if table_match:
            table_html = table_match.group(0)
            # Remove <img> tags if any
            table_html = re.sub(r"<img[^>]*>", "", table_html)
            # Remove markdown code blocks
            table_html = re.sub(r"```html\s*", "", table_html)
            table_html = re.sub(r"```\s*", "", table_html)
            return table_html.strip()

        # Try markdown table
        if "|" in content and "\n" in content:
            markdown_html = self._convert_markdown_to_html(content)
            if markdown_html != "<table></table>":
                return markdown_html

        # Try to parse as plain text table
        return self._convert_text_to_html(content)

    def _convert_markdown_to_html(self, content: str) -> str:
        """Convert markdown table to HTML."""
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        # Find table lines (contain |)
        table_lines = [line for line in lines if "|" in line]

        if not table_lines:
            return "<table></table>"

        html_rows = []
        for idx, line in enumerate(table_lines):
            # Skip separator lines (|----|-----|)
            if re.match(r"^\|[\s\-:]+\|$", line):
                continue

            # Split by | and clean
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]

            if cells:
                # First row (or before separator) is header
                tag = "th" if idx == 0 else "td"
                cells_html = "".join(
                    f"<{tag}>{self._escape_html(cell)}</{tag}>" for cell in cells
                )
                html_rows.append(f"<tr>{cells_html}</tr>")

        if not html_rows:
            return "<table></table>"

        return f"<table>{''.join(html_rows)}</table>"

    def _convert_text_to_html(self, content: str) -> str:
        """
        Convert plain text table to HTML.
        Attempts to detect tabular structure from plain text.
        First row is treated as header (uses <th> tags).
        """
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        if not lines:
            return "<table></table>"

        html_rows = []
        for idx, line in enumerate(lines):
            # Try to detect cells by common separators
            # Check for tab, multiple spaces, or pipe separators
            if "\t" in line:
                cells = [cell.strip() for cell in line.split("\t")]
            elif "  " in line:  # Multiple spaces
                cells = [cell.strip() for cell in re.split(r"\s{2,}", line)]
            elif "|" in line:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            else:
                # Single cell row
                cells = [line]

            if cells:
                # First row uses <th> (headers), subsequent rows use <td>
                tag = "th" if idx == 0 else "td"
                cells_html = "".join(
                    f"<{tag}>{self._escape_html(cell)}</{tag}>" for cell in cells
                )
                html_rows.append(f"<tr>{cells_html}</tr>")

        if not html_rows:
            return "<table></table>"

        return f"<table>{''.join(html_rows)}</table>"

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @modal.method()
    def process_batch(self, images_data: List[Dict]) -> List[Dict]:
        """
        Process a batch of images.

        Args:
            images_data: List of dicts with 'filename' and 'image_pil'

        Returns:
            List of result dicts
        """
        from PIL import Image

        batch_start = time.time()

        # Convert base64 images to PIL Images
        pil_images_data = []
        for img_data in images_data:
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(img_data["image_base64"])
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                pil_images_data.append(
                    {"filename": img_data["filename"], "image_pil": image_pil}
                )
            except Exception as e:
                print(f"  ✗ Failed to decode image {img_data['filename']}: {e}")
                pil_images_data.append(
                    {
                        "filename": img_data["filename"],
                        "image_pil": None,
                        "error": str(e),
                    }
                )

        # Process images sequentially (vLLM batches internally)
        results = []
        for img_data in pil_images_data:
            if img_data.get("image_pil") is None:
                results.append(
                    {
                        "filename": img_data["filename"],
                        "success": False,
                        "table_html": "<table></table>",
                        "raw_content": "",
                        "processing_time": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "model": MODEL_NAME,
                        "error": img_data.get("error", "Failed to decode image"),
                    }
                )
            else:
                result = self._process_single_image(img_data)
                results.append(result)

        batch_time = time.time() - batch_start
        successful = sum(1 for r in results if r["success"])

        print(
            f"  Batch completed: {successful}/{len(results)} successful, {batch_time:.2f}s total"
        )

        return results


def load_all_pdfs(pdf_dir: str = "data/rd-tablebench/pdfs") -> List[Dict]:
    """
    Load all PDFs from the dataset directory and convert to base64-encoded PNG images.

    This runs locally, not on Modal.

    Args:
        pdf_dir: Directory containing PDFs

    Returns:
        List of dicts with 'filename', 'pdf_path', and 'image_base64'
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise ValueError(f"Directory {pdf_dir} does not exist")

    # Get all PDF files
    pdf_files = sorted(list(pdf_path.glob("*.pdf")))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF files to process")
    print()

    images_data = []
    errors = []

    for idx, file_path in enumerate(pdf_files):
        try:
            # Open PDF and get first page
            doc = fitz.open(str(file_path))
            page = doc[0]

            # Render to image at specified DPI
            pix = page.get_pixmap(dpi=PDF_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()

            # Downsample if too large
            MAX_PIXELS = 50_000_000
            current_pixels = img.width * img.height
            if current_pixels > MAX_PIXELS:
                scale = (MAX_PIXELS / current_pixels) ** 0.5
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to base64 PNG
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            img_b64 = base64.b64encode(img_bytes.read()).decode("utf-8")

            images_data.append(
                {
                    "filename": file_path.name,
                    "pdf_path": str(file_path),
                    "image_base64": img_b64,
                }
            )

            # Progress indicator every 100 files
            if (idx + 1) % 100 == 0:
                print(f"  Loaded {idx + 1}/{len(pdf_files)} PDFs...")

        except Exception as e:
            error_msg = f"  ✗ {file_path.name}: {e}"
            print(error_msg)
            errors.append(error_msg)

    print(f"✓ Successfully loaded {len(images_data)}/{len(pdf_files)} PDFs")

    if errors:
        print(f"⚠ {len(errors)} files failed to load")

    return images_data


def save_results(results: List[Dict], output_dir: Path) -> Dict:
    """
    Save HTML and JSON results locally.

    Args:
        results: List of result dicts from processing
        output_dir: Directory to save results

    Returns:
        Dict with summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {output_dir}...")

    success_count = 0
    error_count = 0
    processing_times = []
    total_tokens = []

    for result in results:
        filename = result["filename"]
        basename = Path(filename).stem

        # Save table HTML for grading
        html_filename = filename.replace(".pdf", ".html")
        html_path = output_dir / html_filename
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(result["table_html"])

        # Save metadata as JSON
        json_path = output_dir / f"{basename}.json"
        metadata = {
            "filename": result["filename"],
            "success": result["success"],
            "processing_time": result["processing_time"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "model": result["model"],
            "error": result["error"],
            "raw_content": result.get("raw_content", "")[
                :1000
            ],  # First 1000 chars for debugging
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Track statistics
        if result["success"]:
            success_count += 1
            processing_times.append(result["processing_time"])
            total_tokens.append(result["total_tokens"])
            print(f"  ✓ {filename}")
        else:
            error_count += 1
            print(f"  ✗ {filename}: {result['error']}")

    # Calculate summary statistics
    summary = {
        "total_files": len(results),
        "successful": success_count,
        "failed": error_count,
        "success_rate": success_count / len(results) * 100 if results else 0,
        "total_processing_time": sum(processing_times),
        "avg_processing_time": (
            statistics.mean(processing_times) if processing_times else 0
        ),
        "median_processing_time": (
            statistics.median(processing_times) if processing_times else 0
        ),
        "total_tokens": sum(total_tokens),
        "avg_tokens_per_file": statistics.mean(total_tokens) if total_tokens else 0,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "gpu": GPU_TYPE,
            "prompt": PROMPT,
        },
    }

    # Save summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total files:              {summary['total_files']}")
    print(f"Successful:               {summary['successful']}")
    print(f"Failed:                   {summary['failed']}")
    print(f"Success rate:             {summary['success_rate']:.2f}%")
    print(f"Total processing time:    {summary['total_processing_time']:.2f}s")
    print(f"Avg processing time:      {summary['avg_processing_time']:.2f}s")
    print(f"Median processing time:   {summary['median_processing_time']:.2f}s")
    print(f"Total tokens:             {summary['total_tokens']:,}")
    print(f"Avg tokens per file:      {summary['avg_tokens_per_file']:.0f}")
    print("=" * 80)

    return summary


@app.local_entrypoint()
def main():
    """Main entry point for the benchmark."""
    print("=" * 80)
    print("DeepSeek-OCR VLLM Benchmark - Full Reducto TableBench")
    print("=" * 80)
    print("Configuration:")
    print(f"  Model:                {MODEL_NAME}")
    print(f"  GPU:                  {GPU_TYPE}")
    print("  Parallel containers:  10")
    print(f"  batch_size:           {BATCH_SIZE} per container")
    print(f"  Prompt:               {PROMPT}")
    print("=" * 80)
    print()

    # Step 1: Load all PDFs locally
    print("[Step 1/3] Loading PDFs and converting to images...")
    start_time = time.time()
    images_data = load_all_pdfs()
    load_time = time.time() - start_time
    print(f"✓ Loading completed in {load_time:.2f}s")
    print()

    if not images_data:
        print("ERROR: No images to process!")
        return

    # Step 2: Process all PDFs in batches on Modal with parallel containers
    print(
        f"[Step 2/3] Processing {len(images_data)} PDFs with 10 parallel GPU containers..."
    )
    num_batches = (len(images_data) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches: {num_batches}")
    print(f"Batch size per container: {BATCH_SIZE}")
    print()

    process_start = time.time()

    # Split into batches for parallel processing
    batches = []
    for batch_idx in range(0, len(images_data), BATCH_SIZE):
        batch = images_data[batch_idx : batch_idx + BATCH_SIZE]
        batches.append(batch)

    print(f"Processing {len(batches)} batches across 10 parallel containers...")
    print("Expected speedup: ~10x faster than sequential processing")
    print()

    # Process batches in parallel using .map()
    # Modal will spawn up to 10 containers running simultaneously
    all_results = []
    batch_num = 0
    for batch_results in DeepSeekOCRProcessor().process_batch.map(
        batches, order_outputs=False
    ):
        all_results.extend(batch_results)
        batch_num += 1
        completed = len(all_results)
        successful = sum(1 for r in batch_results if r["success"])
        print(
            f"  Batch {batch_num}/{len(batches)}: {successful}/{len(batch_results)} successful | Total: {completed}/{len(images_data)} PDFs"
        )

    process_time = time.time() - process_start
    throughput = len(images_data) / process_time if process_time > 0 else 0

    print()
    print("=" * 80)
    print("✓ All Documents Processed!")
    print("=" * 80)
    print(f"Processing time:  {process_time:.2f}s ({process_time / 60:.2f} minutes)")
    print(f"Throughput:       {throughput:.2f} images/sec")
    print("=" * 80)
    print()

    # Step 3: Save results locally
    print("[Step 3/3] Saving results...")
    output_dir = Path("data/rd-tablebench/providers/deepseek-ocr")
    save_results(all_results, output_dir)

    print()
    print("=" * 80)
    print("✓ Benchmark Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Run grading: uv run python grading.py --model deepseek-ocr")
    print(f"  2. View summary: cat {output_dir}/benchmark_summary.json")
    print("=" * 80)


if __name__ == "__main__":
    # Note: This script is designed to run with Modal
    # Run with: modal run ocr_benchmark_deepseek.py
    pass
