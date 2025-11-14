#!/usr/bin/env python3

import base64
import concurrent.futures
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
app = modal.App("ocr-benchmark-dots")

# Constants
MODEL_NAME = "rednote-hilab/dots.ocr"
MAX_NUM_SEQS = 128
BATCH_SIZE = 8
MAX_NUM_BATCHED_TOKENS = 16384
# Use table-specific prompt
PROMPT = "Table Recognition:"
GPU_TYPE = "A100-80GB"
TIMEOUT = 7200  # 2 hours
PDF_DPI = 200
MAX_RETRIES = 3
VLLM_PORT = 8000

# VLLM image optimized for DOTS OCR with model pre-downloading
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.8.0+cu129",
        "torchvision==0.23.0+cu129",
        "torchaudio==2.8.0+cu129",
        index_url="https://download.pytorch.org/whl/cu129",  # CUDA 12.9
    )
    .run_commands(
        # Install vLLM 0.11.0+ with official DOTS OCR support
        "pip install vllm>=0.11.0"
    )
    .pip_install(
        "openai",
        "Pillow",
        "PyMuPDF",
        "huggingface-hub",
    )
    .run_commands(
        # Pre-download the model to avoid network issues during runtime
        f"huggingface-cli download {MODEL_NAME} --local-dir /models/huggingface/hub/dots-ocr",
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
class VLLMServer:
    """VLLM Server for DOTS OCR with optimal throughput configuration."""

    @modal.enter()
    def start_server(self):
        """Start the VLLM server with optimal configuration."""
        import subprocess
        import threading
        import time

        self.model_name = MODEL_NAME
        self.port = VLLM_PORT
        self.server_process = None
        self.server_logs = []

        print("=" * 80)
        print(f"Starting VLLM server for DOTS OCR with max_num_seqs={MAX_NUM_SEQS}")
        print("=" * 80)

        # Build VLLM command with optimal configuration
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--trust-remote-code",
            "--max-num-batched-tokens",
            str(MAX_NUM_BATCHED_TOKENS),
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--gpu-memory-utilization",
            "0.95",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--download-dir",
            "/models/huggingface",
        ]

        def log_output(pipe, prefix):
            """Capture server output."""
            for line in iter(pipe.readline, ""):
                if line:
                    log_line = f"[{prefix}] {line.rstrip()}"
                    self.server_logs.append(log_line)
                    print(log_line)

        # Start server in background
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Start threads to capture output
        stdout_thread = threading.Thread(
            target=log_output, args=(self.server_process.stdout, "VLLM-OUT")
        )
        stderr_thread = threading.Thread(
            target=log_output, args=(self.server_process.stderr, "VLLM-ERR")
        )
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Wait for server to be ready
        print("Waiting for server to be ready...")
        max_wait = 600  # 10 minutes (model already downloaded)
        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Check if process died
            if self.server_process.poll() is not None:
                print(
                    f"ERROR: VLLM server process exited with code {self.server_process.returncode}"
                )
                print("\nRecent logs:")
                for log in self.server_logs[-50:]:
                    print(log)
                raise RuntimeError("VLLM server process terminated unexpectedly")

            try:
                import requests

                response = requests.get(
                    f"http://localhost:{self.port}/health", timeout=5
                )
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"✓ VLLM server is ready! (took {elapsed:.1f}s)")
                    print("=" * 80)
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            print("\nERROR: Server failed to start within timeout")
            print("\nRecent logs:")
            for log in self.server_logs[-50:]:
                print(log)
            raise RuntimeError("VLLM server failed to start within timeout")

    @modal.exit()
    def stop_server(self):
        """Stop the VLLM server."""
        if self.server_process:
            print("Stopping VLLM server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("✓ Server stopped")

    def _process_single_image(
        self, client, img_data: Dict, retry_count: int = 0
    ) -> Dict:
        """
        Process a single image with retry logic.

        Args:
            client: OpenAI client instance
            img_data: Dict with 'filename' and 'image_base64'
            retry_count: Current retry attempt

        Returns:
            Dict with results including table HTML, processing time, and any errors
        """
        filename = img_data["filename"]
        image_b64 = img_data["image_base64"]

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]

            start_time = time.time()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=16384,
            )
            latency = time.time() - start_time

            # Extract response content
            content = response.choices[0].message.content

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

            result = {
                "filename": filename,
                "success": True,
                "table_html": table_html,
                "raw_content": content,
                "processing_time": latency,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
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
            ):
                wait_time = 2**retry_count  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"  ⚠ {filename}: Retry {retry_count + 1}/{MAX_RETRIES} after {wait_time}s - {error_msg}"
                )
                time.sleep(wait_time)
                return self._process_single_image(client, img_data, retry_count + 1)

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
        Extract and convert table from DOTS OCR response to HTML.

        DOTS OCR may return:
        1. JSON with layout information including tables
        2. Direct HTML table
        3. Markdown table
        4. Plain text table

        Args:
            content: Raw response content from VLLM

        Returns:
            HTML table string
        """
        # First, try to parse as JSON (DOTS OCR layout format)
        try:
            layout_data = json.loads(content)
            if isinstance(layout_data, dict) or isinstance(layout_data, list):
                return self._convert_dots_layout_to_html(layout_data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Look for existing HTML table tags
        table_match = re.search(
            r"<table[^>]*>.*?</table>", content, re.DOTALL | re.IGNORECASE
        )

        if table_match:
            table_html = table_match.group(0)
            # Remove <img> tags
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

    def _convert_dots_layout_to_html(self, layout_data) -> str:
        """
        Convert DOTS OCR layout JSON to HTML table.

        DOTS OCR outputs structured layout information with bboxes and text.
        We need to find table structures and convert them to HTML.

        Args:
            layout_data: Parsed JSON layout data

        Returns:
            HTML table string
        """
        # Handle list of layout elements
        if isinstance(layout_data, list):
            # Look for table elements
            for element in layout_data:
                if isinstance(element, dict):
                    # Check if this is a table element
                    element_type = element.get("type", "").lower()
                    if "table" in element_type:
                        # Extract table structure
                        return self._parse_table_element(element)

            # If no explicit table found, try to construct from all elements
            return self._construct_table_from_elements(layout_data)

        # Handle dict layout
        elif isinstance(layout_data, dict):
            # Look for table key
            if "table" in layout_data:
                return self._parse_table_element(layout_data["table"])

            # Look for cells or structured data
            if "cells" in layout_data:
                return self._construct_table_from_cells(layout_data["cells"])

            # If it has bbox and text, treat as single cell
            if "text" in layout_data:
                return f"<table><tr><td>{self._escape_html(layout_data['text'])}</td></tr></table>"

        return "<table></table>"

    def _parse_table_element(self, table_element: Dict) -> str:
        """Parse a table element from DOTS OCR layout."""
        # Look for cells, rows, or structured content
        if "cells" in table_element:
            return self._construct_table_from_cells(table_element["cells"])

        if "rows" in table_element:
            return self._construct_table_from_rows(table_element["rows"])

        if "html" in table_element:
            return table_element["html"]

        if "text" in table_element:
            # Try to parse text as table
            return self._convert_text_to_html(table_element["text"])

        return "<table></table>"

    def _construct_table_from_cells(self, cells: List) -> str:
        """Construct HTML table from cell list."""
        if not cells:
            return "<table></table>"

        # Sort cells by position (top to bottom, left to right)
        sorted_cells = sorted(cells, key=lambda c: (c.get("row", 0), c.get("col", 0)))

        # Group by rows
        rows = {}
        for cell in sorted_cells:
            row_idx = cell.get("row", 0)
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(cell)

        # Build HTML
        html_rows = []
        for row_idx in sorted(rows.keys()):
            row_cells = rows[row_idx]
            # First row uses <th>, others use <td>
            tag = "th" if row_idx == 0 else "td"
            cells_html = "".join(
                f"<{tag}>{self._escape_html(cell.get('text', ''))}</{tag}>"
                for cell in row_cells
            )
            html_rows.append(f"<tr>{cells_html}</tr>")

        if not html_rows:
            return "<table></table>"

        return f"<table>{''.join(html_rows)}</table>"

    def _construct_table_from_rows(self, rows: List) -> str:
        """Construct HTML table from row list."""
        if not rows:
            return "<table></table>"

        html_rows = []
        for idx, row in enumerate(rows):
            if isinstance(row, list):
                # Row is list of cells
                tag = "th" if idx == 0 else "td"
                cells_html = "".join(
                    f"<{tag}>{self._escape_html(str(cell))}</{tag}>" for cell in row
                )
                html_rows.append(f"<tr>{cells_html}</tr>")
            elif isinstance(row, dict) and "cells" in row:
                # Row has cells key
                tag = "th" if idx == 0 else "td"
                cells_html = "".join(
                    f"<{tag}>{self._escape_html(str(cell))}</{tag}>"
                    for cell in row["cells"]
                )
                html_rows.append(f"<tr>{cells_html}</tr>")

        if not html_rows:
            return "<table></table>"

        return f"<table>{''.join(html_rows)}</table>"

    def _construct_table_from_elements(self, elements: List) -> str:
        """Construct table from generic layout elements based on spatial arrangement."""
        if not elements:
            return "<table></table>"

        # Extract elements with bbox and text
        text_elements = []
        for elem in elements:
            if isinstance(elem, dict) and "bbox" in elem and "text" in elem:
                bbox = elem["bbox"]
                if len(bbox) >= 4:
                    text_elements.append(
                        {
                            "text": elem["text"],
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                        }
                    )

        if not text_elements:
            return "<table></table>"

        # Sort by y position to group into rows
        text_elements.sort(key=lambda e: e["y1"])

        # Group elements into rows based on y position
        rows = []
        current_row = []
        current_y = text_elements[0]["y1"]
        y_threshold = 10  # pixels

        for elem in text_elements:
            if abs(elem["y1"] - current_y) < y_threshold:
                current_row.append(elem)
            else:
                if current_row:
                    # Sort row by x position
                    current_row.sort(key=lambda e: e["x1"])
                    rows.append(current_row)
                current_row = [elem]
                current_y = elem["y1"]

        if current_row:
            current_row.sort(key=lambda e: e["x1"])
            rows.append(current_row)

        # Build HTML table
        html_rows = []
        for idx, row in enumerate(rows):
            tag = "th" if idx == 0 else "td"
            cells_html = "".join(
                f"<{tag}>{self._escape_html(cell['text'])}</{tag}>" for cell in row
            )
            html_rows.append(f"<tr>{cells_html}</tr>")

        if not html_rows:
            return "<table></table>"

        return f"<table>{''.join(html_rows)}</table>"

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
        Process a batch of images concurrently.

        Each container processes BATCH_SIZE images concurrently.
        Up to 10 containers run in parallel via .map() concurrency.

        Args:
            images_data: List of dicts with 'filename' and 'image_base64'

        Returns:
            List of result dicts
        """
        from openai import OpenAI

        # Create OpenAI client
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{self.port}/v1",
            timeout=3600,
        )

        batch_start = time.time()

        # Process images concurrently within batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [
                executor.submit(self._process_single_image, client, img)
                for img in images_data
            ]
            results = [f.result() for f in futures]

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
                img = img.resize(new_size, Image.LANCZOS)

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
            "max_num_seqs": MAX_NUM_SEQS,
            "batch_size": BATCH_SIZE,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu": GPU_TYPE,
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
    print("DOTS OCR VLLM Benchmark - Full Reducto TableBench")
    print("=" * 80)
    print("Configuration:")
    print(f"  Model:                {MODEL_NAME}")
    print(f"  GPU:                  {GPU_TYPE}")
    print("  Parallel containers:  10")
    print(f"  max_num_seqs:         {MAX_NUM_SEQS} per container")
    print(f"  batch_size:           {BATCH_SIZE} per container")
    print(f"  max_num_batched_tokens: {MAX_NUM_BATCHED_TOKENS}")
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
    for batch_results in VLLMServer().process_batch.map(batches, order_outputs=False):
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
    output_dir = Path("data/rd-tablebench/providers/dots-ocr")
    save_results(all_results, output_dir)

    print()
    print("=" * 80)
    print("✓ Benchmark Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Run grading: uv run python grading.py --model dots-ocr")
    print(f"  2. View summary: cat {output_dir}/benchmark_summary.json")
    print("=" * 80)


if __name__ == "__main__":
    # Note: This script is designed to run with Modal
    # Run with: modal run ocr_benchmark_dots.py
    pass
