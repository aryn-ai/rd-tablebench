import os
import json
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
from aryn_sdk.partition import partition_file_async_result

load_dotenv()
ARYN_API_KEY = os.environ["ARYN_TEST_API_KEY"]

PROVIDER = "detr-5.0_tatr"
PARTITIONED_ROOT = Path("partitioned")
OUTPUT_DIR = PARTITIONED_ROOT / PROVIDER
TASKS_DIR = Path("tasks")
ASYNC_RESULT_URL = "https://test-api.aryn.ai/v1/async/result"

def load_tasks(tasks_file_path):
    with open(tasks_file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [(row[0], row[1]) for row in reader]

def is_partitioned(json_path):
    if not json_path.exists():
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        return 'elements' in data

def process_task(filename, task_id, output_path):
    try:
        response = partition_file_async_result(
            task_id=task_id,
            aryn_api_key=ARYN_API_KEY,
            async_result_url=ASYNC_RESULT_URL
        )
        
        if response["task_status"] == "done":
            with open(output_path, "w") as f:
                json.dump(response["result"], f)
            print(f"Wrote result to {output_path}")
            return True
        else:
            print(f"Task {task_id} status: {response['task_status']}")
            return False
            
    except Exception as e:
        print(f"Error processing task {task_id}, filename: {filename}: {e}")
        return False

def main():
    if not TASKS_DIR.exists():
        print(f"Tasks folder {TASKS_DIR} does not exist")
        return
    
    PARTITIONED_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks_file = TASKS_DIR / f"tasks_{PROVIDER}.csv"
    tasks = load_tasks(tasks_file)
    
    for filename, task_id in tasks:
        output_path = OUTPUT_DIR / f"{filename}.json"
        
        if is_partitioned(output_path):
            print(f"Skipping {filename} because it already exists")
            continue
            
        print(f'Processing: {task_id}')
        time.sleep(0.1)  # Rate limiting
        process_task(filename, task_id, output_path)

if __name__ == "__main__":
    main() 