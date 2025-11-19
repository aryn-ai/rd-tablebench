import glob
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import backoff
from google.genai import Client
from google.genai import types
from pdf2image import convert_from_path
from tqdm import tqdm

from providers.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model to use for table extraction
MODEL_NAME = "gemini-3-pro-preview"

# Standard table extraction prompt
TABLE_EXTRACTION_PROMPT = (
    "Convert the image to an HTML table. The output should begin with <table> "
    "and end with </table>. Specify rowspan and colspan attributes when they "
    "are greater than 1. Do not specify any other attributes. Only use table "
    "related HTML tags, no additional formatting is required."
)

# Initialize paths
base_path = os.path.expanduser(settings.input_dir)
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)

# Initialize Gemini client
assert settings.gemini_api_key, "GEMINI_API_KEY must be set in environment"
client = Client(api_key=settings.gemini_api_key)


def convert_pdf_to_png_bytes(pdf_path: str) -> bytes:
    logger.debug(f"Converting PDF to PNG: {os.path.basename(pdf_path)}")
    images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
    img_buffer = io.BytesIO()
    images[0].save(img_buffer, format="PNG")
    logger.debug(f"Conversion complete: {os.path.basename(pdf_path)}")
    return img_buffer.getvalue()


def is_retryable_error(exception: Exception) -> bool:
    error_str = str(exception).lower()
    return any(
        keyword in error_str
        for keyword in ["rate", "quota", "timeout", "503", "429", "500"]
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    giveup=lambda e: not is_retryable_error(e),
    max_tries=5,
    max_time=300,  # 5 minutes max
)
def analyze_document_with_gemini(image_bytes: bytes) -> tuple[str, dict[str, Any]]:
    logger.debug(
        f"Sending request to Gemini API (image size: {len(image_bytes)} bytes)"
    )

    # Create content with text prompt and image
    content = types.Content(
        parts=[
            types.Part.from_text(text=TABLE_EXTRACTION_PROMPT),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
        role="user",
    )

    # Configure generation
    config = types.GenerateContentConfig(
        temperature=0,
        candidate_count=1,
    )

    # Generate response
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=content,
        config=config,
    )

    # Extract metadata
    usage_metadata = {}
    if response.usage_metadata:
        md = response.usage_metadata
        usage_metadata = {
            "prompt_tokens": int(md.prompt_token_count) if md.prompt_token_count else 0,
            "completion_tokens": (
                int(md.candidates_token_count) if md.candidates_token_count else 0
            ),
            "total_tokens": int(md.total_token_count) if md.total_token_count else 0,
        }
        logger.debug(f"API response received: {usage_metadata['total_tokens']} tokens")

    # Extract text from response
    if (
        not response.candidates
        or not response.candidates[0].content
        or not response.candidates[0].content.parts
    ):
        raise ValueError("No content in Gemini response")

    # Combine all text parts
    output_text = " ".join(
        part.text if part and part.text else ""
        for part in response.candidates[0].content.parts
    )

    return output_text, usage_metadata


def extract_html_table(content: str) -> str | None:
    start = content.find("<table>")
    end = content.find("</table>")

    if start != -1 and end != -1:
        return content[start : end + 8]

    return None


def process_pdf(pdf_path: str, force: bool = False) -> tuple[str, str | None, bool]:
    filename = os.path.basename(pdf_path)

    # Create output paths
    output_html_path = pdf_path.replace("pdfs", f"outputs/{MODEL_NAME}").replace(
        ".pdf", ".html"
    )
    output_json_path = output_html_path.replace(".html", ".json")

    # Check if already processed
    if (
        not force
        and os.path.exists(output_html_path)
        and os.path.exists(output_json_path)
    ):
        logger.debug(f"Skipping already processed file: {filename}")
        return pdf_path, None, True

    logger.info(f"Processing: {filename}")
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    start_time = time.time()

    try:
        # Convert PDF to image
        image_bytes = convert_pdf_to_png_bytes(pdf_path)

        # Send to Gemini API
        response_text, usage_metadata = analyze_document_with_gemini(image_bytes)

        # Extract HTML table
        html_table = extract_html_table(response_text)

        processing_time = time.time() - start_time

        # Prepare result metadata
        result = {
            "filename": filename,
            "success": html_table is not None,
            "table_html": html_table if html_table else response_text,
            "processing_time": processing_time,
            **usage_metadata,
        }

        if not html_table:
            result["error"] = "No HTML table found in response"
            logger.warning(f"No HTML table found in response for {filename}")
        else:
            logger.info(
                f"Successfully extracted table from {filename} ({processing_time:.2f}s, {usage_metadata.get('total_tokens', 0)} tokens)"
            )

        # Save HTML output
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_table if html_table else response_text)

        # Save JSON metadata
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return pdf_path, None, False

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(
            f"Failed to process {filename}: {error_msg} ({processing_time:.2f}s)"
        )

        # Save error metadata
        result = {
            "filename": filename,
            "success": False,
            "table_html": None,
            "processing_time": processing_time,
            "error": error_msg,
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return pdf_path, error_msg, False


def process_all_pdfs(pdfs: list[str], max_workers: int = 10, force: bool = False):
    logger.info(
        f"Starting batch processing of {len(pdfs)} PDFs with {max_workers} workers"
    )
    if not force:
        logger.info("Skipping already processed files (use --force to reprocess)")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf, force): pdf for pdf in pdfs}

        progress_bar = tqdm(total=len(pdfs), desc=f"Processing PDFs with {MODEL_NAME}")

        errors = []
        successes = 0
        skipped = 0
        for future in as_completed(futures):
            pdf_path, error, was_skipped = future.result()
            if was_skipped:
                skipped += 1
            elif error:
                errors.append((pdf_path, error))
            else:
                successes += 1
            progress_bar.update(1)

        progress_bar.close()

    total_time = time.time() - start_time
    processed_count = successes + len(errors)

    logger.info(f"\n{'='*60}")
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Total PDFs: {len(pdfs)}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"  - Successful: {successes}")
    logger.info(f"  - Failed: {len(errors)}")
    logger.info(f"Total time: {total_time:.2f}s")
    if processed_count > 0:
        logger.info(
            f"Average time per processed PDF: {total_time/processed_count:.2f}s"
        )

    if errors:
        logger.error(f"\nErrors encountered in {len(errors)} PDFs:")
        for pdf_path, error in errors[:10]:  # Show first 10 errors
            logger.error(f"  - {os.path.basename(pdf_path)}: {error}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
    else:
        if processed_count > 0:
            logger.info("All processed PDFs completed successfully!")
        if skipped > 0:
            logger.info(f"Skipped {skipped} already processed PDFs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract tables from PDFs using Gemini 3 Pro Preview"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if output already exists",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Gemini Table Extraction")
    logger.info("=" * 60)
    logger.info(f"Found {len(pdfs)} PDFs to process")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Force reprocess: {args.force}")
    logger.info(f"Input directory: {base_path}")
    logger.info(f"Output directory: outputs/{MODEL_NAME}")
    logger.info("=" * 60)

    process_all_pdfs(pdfs, max_workers=args.num_workers, force=args.force)
