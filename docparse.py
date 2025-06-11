import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv


from aryn_sdk.partition import partition_file_async_submit

load_dotenv()
ARYN_API_KEY = os.environ["ARYN_TEST_API_KEY"]

PROVIDER = "detr-5.0_deformable"
INPUT_DIR = Path("data/rd-tablebench/pdfs")
OUTPUT_ROOT = Path("partitioned")
DOCPARSE_URL = "https://test-api.aryn.ai/v1/document/partition"

TASKS_DIR = Path("tasks")

LANGUAGE_MAP = {
    'en': 'english', 'ru': 'russian', 'de': 'german', 'es': 'spanish',
    'fr': 'french', 'ja': 'japanese', 'zh': 'chinese', 'it': 'italian',
    'pt': 'portuguese', 'uk': 'ukrainian', 'nl': 'dutch', 'cs': 'czech',
    'pl': 'polish', 'sv': 'swedish', 'ro': 'romanian'
}

def load_score_map():
    with open("score_map.json") as f:
        return json.load(f)

def load_tasks(tasks_file_path):
    try:
        with open(tasks_file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return {row[0] for row in reader}
    except FileNotFoundError:
        tasks_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tasks_file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'task_id'])
        return set()

def is_partitioned(output_path):
    if not output_path.exists():
        return False
    
    with open(output_path, 'r') as f:
        data = json.load(f)
        return 'elements' in data

def process_pdf(pdf_file, pdf_path, output_path, language, tasks_file):
    with open(pdf_path, "rb") as f:
        response = partition_file_async_submit(
            f,
            text_mode="vision_ocr",
            extract_table_structure=True,
            ocr_language=language,
            docparse_url=DOCPARSE_URL,
            table_extraction_options={
                "include_additional_text": True,
                "model_selection": "deformable_detr"
            },
            aryn_api_key=ARYN_API_KEY
        )
        
        with open(tasks_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([pdf_file, response['task_id']])

def main():
    output_dir = OUTPUT_ROOT / PROVIDER
    output_dir.mkdir(parents=True, exist_ok=True)
    
    score_map = load_score_map()
    tasks_file = Path(TASKS_DIR / f"tasks_{PROVIDER}.csv")
    sent_files = load_tasks(tasks_file)
    
    for pdf_path in INPUT_DIR.glob("*.pdf"):
        output_path = output_dir / f"{pdf_path.name}.json"
        print(f"Processing {pdf_path.name}")
        
        if output_path.exists() and is_partitioned(output_path):
            print(f"Skipping {pdf_path.name} because it already exists")
            continue
        if pdf_path.name in sent_files:
            print(f"Skipping {pdf_path.name} because it has already been sent")
            continue
            
        language = LANGUAGE_MAP[score_map[pdf_path.name]['language']]
        process_pdf(pdf_path.name, pdf_path, output_path, language, tasks_file)

if __name__ == "__main__":
    main() 