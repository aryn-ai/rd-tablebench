import os
import glob
import json
import asyncio
import aiohttp
import time
from typing import Tuple
from tqdm import tqdm

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)

API_KEY = os.environ.get("CHUNKR_API_KEY")
if not API_KEY:
    raise ValueError("CHUNKR_API_KEY environment variable is not set")

CHUNKR_URL = "https://api.chunkr.ai/api/v1/task"
HEADERS = {"Authorization": API_KEY}


async def process_pdf(
    pdf_path: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> Tuple[str, dict]:
    async with semaphore:
        try:
            with open(pdf_path, "rb") as file:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    file,
                    filename=os.path.basename(pdf_path),
                    content_type="application/pdf",
                )
                data.add_field("model", "HighQuality")
                data.add_field("target_chunk_length", "512")
                data.add_field("ocr_strategy", "Auto")

                async with session.post(
                    CHUNKR_URL, headers=HEADERS, data=data
                ) as response:
                    if response.status == 200:
                        task_info = await response.json()
                        task_id = task_info["task_id"]
                        result = await poll_task(pdf_path, task_id, session)
                        return pdf_path, result
                    else:
                        return pdf_path, {
                            "error": f"Error: {response.status}, {await response.text()}"
                        }
        except Exception as e:
            return pdf_path, {"error": str(e)}


async def poll_task(
    pdf_path: str, task_id: str, session: aiohttp.ClientSession
) -> dict:
    task_url = f"{CHUNKR_URL}/{task_id}"
    start_time = time.time()
    while time.time() - start_time < 60 * 10:
        try:
            async with session.get(task_url, headers=HEADERS) as response:
                if response.status == 200:
                    task_info = await response.json()
                    if task_info["status"] == "Succeeded":
                        return dict(task_info)
                    elif task_info["status"] in ["Failed", "Canceled"]:
                        return {"error": f"Task failed or canceled: {task_info}"}
                else:
                    return {
                        "error": f"Error polling task: {response.status}, {await response.text()}"
                    }
        except Exception as e:
            return {"error": str(e)}
        await asyncio.sleep(5)  # Wait 5 seconds before polling again
    return {"error": "Timeout: Task did not complete within 10 minutes"}


async def process_all_pdfs(pdfs: list[str]):
    semaphore = asyncio.Semaphore(100)  # Limit to 100
    async with aiohttp.ClientSession() as session:
        tasks = [process_pdf(pdf, session, semaphore) for pdf in pdfs]
        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")
        results = []
        for task in asyncio.as_completed(tasks):
            pdf_path, result = await task
            results.append((pdf_path, result))
            progress_bar.update(1)
        progress_bar.close()

    for pdf_path, result in results:
        output_path = pdf_path.replace("pdfs", "chunkr_nov1").replace(".pdf", ".json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"Processed {len(results)} PDFs")


asyncio.run(process_all_pdfs(pdfs))
