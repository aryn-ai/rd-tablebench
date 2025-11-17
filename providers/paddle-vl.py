#!/usr/bin/env python3
import json
import os
import re
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import modal

# Modal app
app = modal.App("ocr-benchmark-paddle-parallel")

# Build Modal image with PaddlePaddle and PaddleOCR (same as original)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "wget",
        # OpenGL libraries required by OpenCV
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .run_commands(
        # Install PaddlePaddle GPU version (CUDA 12.6)
        "python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        # Install PaddleOCR with document parsing support
        "python -m pip install -U 'paddleocr[doc-parser]'",
        # Install safetensors for model loading
        "python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl",
    )
    .pip_install("PyMuPDF")  # For PDF handling
    .env(
        {
            "PADDLEOCR_OFFLINE": "0",
            "PADDLEOCR_HOME": "/models/paddleocr",
        }
    )
)

# Modal volumes for caching models
models_volume = modal.Volume.from_name("ocr-paddle-models", create_if_missing=True)


def init_paddleocr_with_retry(max_retries=3):
    """Initialize PaddleOCR with retry logic for network issues."""
    import time
    from paddleocr import PaddleOCRVL

    for attempt in range(max_retries):
        try:
            print(f"  Initializing PaddleOCR (attempt {attempt + 1}/{max_retries})...")
            pipeline = PaddleOCRVL()
            print("  ✓ Pipeline initialized successfully")
            return pipeline
        except Exception as e:
            error_msg = str(e)
            if (
                "No available model hosting platforms" in error_msg
                or "network" in error_msg.lower()
            ):
                if attempt < max_retries - 1:
                    delay = 5 * (2**attempt)  # Exponential backoff
                    print(f"  ⚠ Network error: {error_msg}")
                    print(f"  ⏳ Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  ✗ Failed after {max_retries} attempts: {error_msg}")
                    raise
            else:
                print(f"  ✗ Initialization error: {error_msg}")
                raise


@app.function(
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/models": models_volume},
    image=image,
)
def process_pdf_batch(batch_data: List[Dict]) -> List[Dict]:
    """
    Process a batch of PDFs through PaddleOCR-VL.

    Args:
        batch_data: List of dicts with 'filename' and 'pdf_data'

    Returns:
        List of result dicts with table HTML and metadata
    """
    import shutil

    # Initialize pipeline once for the batch
    print(f"Initializing PaddleOCR for batch of {len(batch_data)} files...")
    pipeline = init_paddleocr_with_retry(max_retries=3)

    results = []

    for file_info in batch_data:
        filename = file_info["filename"]
        pdf_data = file_info["pdf_data"]

        print(f"Processing: {filename}")

        result = {
            "filename": filename,
            "success": False,
            "table_html": "",
            "processing_time": 0,
            "error": None,
            "model": "PaddleOCR-VL",
        }

        try:
            # Save PDF to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_data)
                tmp_path = tmp.name

            try:
                # Process PDF
                start_time = time.time()
                output = pipeline.predict(tmp_path)
                processing_time = time.time() - start_time

                # Extract results
                if not output or len(output) == 0:
                    raise ValueError("No output from PaddleOCR pipeline")

                # Get first page result
                page_result = output[0]

                # Create temp directory for JSON output
                temp_output_dir = tempfile.mkdtemp()

                # Save to JSON to get structured output
                page_result.save_to_json(save_path=temp_output_dir)
                json_files = list(Path(temp_output_dir).glob("*.json"))

                if not json_files:
                    raise ValueError("Failed to save PaddleOCR output to JSON")

                # Load JSON output
                with open(json_files[0], "r", encoding="utf-8") as f:
                    full_output = json.load(f)

                # Extract table HTML from parsing results
                table_html = "<table></table>"  # Default empty table

                if (
                    "parsing_res_list" in full_output
                    and len(full_output["parsing_res_list"]) > 0
                ):
                    # Find table blocks
                    for block in full_output["parsing_res_list"]:
                        if block.get("block_label") == "table" and block.get(
                            "block_content"
                        ):
                            table_html = block["block_content"]
                            break  # Use first table found

                # Clean the HTML - remove image references
                table_html = re.sub(r"<img[^>]*>", "", table_html)

                result.update(
                    {
                        "success": True,
                        "table_html": table_html,
                        "processing_time": processing_time,
                    }
                )

                print(f"  ✓ {filename}: {processing_time:.2f}s")

                # Clean up temp output directory
                shutil.rmtree(temp_output_dir, ignore_errors=True)

            finally:
                # Clean up temp PDF
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            print(f"  ✗ {filename}: {e}")
            result["error"] = str(e)

        results.append(result)

    return results


def load_all_pdfs(pdf_dir: str = "data/rd-tablebench/pdfs") -> List[Dict]:
    """Load all PDFs from the dataset directory."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise ValueError(f"Directory {pdf_dir} does not exist")

    pdf_files = sorted(list(pdf_path.glob("*.pdf")))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF files to process")

    pdf_data_list = []
    for idx, file_path in enumerate(pdf_files):
        with open(file_path, "rb") as f:
            pdf_data = f.read()

        pdf_data_list.append(
            {
                "filename": file_path.name,
                "pdf_data": pdf_data,
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx + 1}/{len(pdf_files)} PDFs...")

    return pdf_data_list


def save_results(results: List[Dict], output_dir: Path) -> Dict:
    """Save HTML results and generate summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {output_dir}...")

    success_count = 0
    error_count = 0
    processing_times = []

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
            "model": result["model"],
            "error": result["error"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Track statistics
        if result["success"]:
            success_count += 1
            processing_times.append(result["processing_time"])
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
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model": "PaddleOCR-VL",
            "parallel_containers": 10,
            "gpu": "A100-80GB",
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
    print("=" * 80)

    return summary


@app.local_entrypoint()
def main():
    """Main entry point for the benchmark."""
    print("=" * 80)
    print("PaddleOCR-VL Parallel Benchmark - Full Reducto TableBench")
    print("=" * 80)
    print("Configuration:")
    print("  Model:                PaddleOCR-VL (native library)")
    print("  GPU:                  A100-80GB")
    print("  Parallel containers:  10")
    print("  Batch size:           100 PDFs per container")
    print("=" * 80)
    print()

    # Step 1: Load all PDFs locally
    print("[Step 1/3] Loading PDFs...")
    start_time = time.time()
    pdf_data_list = load_all_pdfs()
    load_time = time.time() - start_time
    print(f"✓ Loading completed in {load_time:.2f}s")
    print()

    if not pdf_data_list:
        print("ERROR: No PDFs to process!")
        return

    # Step 2: Process PDFs in parallel batches
    print(
        f"[Step 2/3] Processing {len(pdf_data_list)} PDFs with 10 parallel containers..."
    )
    batch_size = 100  # 1000 PDFs / 10 containers = 100 per container
    print(f"Batch size: {batch_size} PDFs per container")
    print()

    process_start = time.time()

    # Split into batches
    batches = []
    for i in range(0, len(pdf_data_list), batch_size):
        batch = pdf_data_list[i : i + batch_size]
        batches.append(batch)

    print(f"Processing {len(batches)} batches in parallel...")
    print()

    # Process batches in parallel using .map()
    all_results = []
    for batch_idx, batch_results in enumerate(
        process_pdf_batch.map(batches, order_outputs=False)
    ):
        all_results.extend(batch_results)
        completed = len(all_results)
        successful = sum(1 for r in batch_results if r["success"])
        print(
            f"  Batch {batch_idx + 1}/{len(batches)}: {successful}/{len(batch_results)} successful | Total: {completed}/{len(pdf_data_list)} PDFs"
        )

    process_time = time.time() - process_start
    throughput = len(pdf_data_list) / process_time if process_time > 0 else 0

    print()
    print("=" * 80)
    print("✓ All Documents Processed!")
    print("=" * 80)
    print(f"Processing time:  {process_time:.2f}s ({process_time / 60:.2f} minutes)")
    print(f"Throughput:       {throughput:.2f} PDFs/sec")
    print("=" * 80)
    print()

    # Step 3: Save results locally
    print("[Step 3/3] Saving results...")
    output_dir = Path("data/rd-tablebench/providers/paddleocr-vl-parallel")
    save_results(all_results, output_dir)

    print()
    print("=" * 80)
    print("✓ Benchmark Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Run grading: uv run python grade_cli.py --model paddleocr-vl-parallel")
    print(f"  2. View summary: cat {output_dir}/benchmark_summary.json")
    print("=" * 80)


if __name__ == "__main__":
    # Run with: modal run ocr_benchmark_paddle_parallel.py
    pass
