import os
import json
import csv
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from aryn_sdk.partition import partition_file_async_result

load_dotenv()
ARYN_API_KEY = os.environ["ARYN_TEST_API_KEY"]

PROVIDER = "detr-5.0_deformable"
PARTITIONED_ROOT = Path("partitioned")
OUTPUT_DIR = PARTITIONED_ROOT / PROVIDER
TASKS_DIR = Path("tasks")
ASYNC_RESULT_URL = "https://test-api.aryn.ai/v1/async/result"
RETRY_SLEEP_SECONDS = 30  # Time to wait between retry attempts


def load_tasks(tasks_file_path):
    with open(tasks_file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [(row[0], row[1]) for row in reader]


def is_partitioned(json_path):
    if not json_path.exists():
        return False

    with open(json_path, "r") as f:
        data = json.load(f)
        return "elements" in data


def process_task(filename, task_id, output_path):
    try:
        response = partition_file_async_result(
            task_id=task_id,
            aryn_api_key=ARYN_API_KEY,
            async_result_url=ASYNC_RESULT_URL,
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
    parser = argparse.ArgumentParser(
        description="Fetch async partition results and retry until all tasks complete"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable retry mode (process tasks once and exit)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=RETRY_SLEEP_SECONDS,
        help=f"Seconds to wait between retry attempts (default: {RETRY_SLEEP_SECONDS})",
    )
    args = parser.parse_args()

    retry_enabled = not args.no_retry

    if not TASKS_DIR.exists():
        print(f"Tasks folder {TASKS_DIR} does not exist")
        return

    PARTITIONED_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks_file = TASKS_DIR / f"tasks_{PROVIDER}.csv"
    all_tasks = load_tasks(tasks_file)

    attempt = 1
    while True:
        print(f"\n{'=' * 60}")
        print(f"Attempt {attempt}")
        print(f"{'=' * 60}")

        incomplete_tasks = []
        completed_count = 0
        skipped_count = 0

        for filename, task_id in all_tasks:
            output_path = OUTPUT_DIR / f"{filename}.json"

            if is_partitioned(output_path):
                skipped_count += 1
                continue

            print(f"Processing: {task_id}")
            time.sleep(0.1)  # Rate limiting

            if process_task(filename, task_id, output_path):
                completed_count += 1
            else:
                incomplete_tasks.append((filename, task_id))

        total_tasks = len(all_tasks)
        total_complete = total_tasks - len(incomplete_tasks)

        print(f"\n{'=' * 60}")
        print(f"Summary for attempt {attempt}:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Already completed: {skipped_count}")
        print(f"  Newly completed: {completed_count}")
        print(f"  Incomplete: {len(incomplete_tasks)}")
        print(
            f"  Overall completion: {total_complete}/{total_tasks} ({100 * total_complete / total_tasks:.1f}%)"
        )
        print(f"{'=' * 60}")

        if len(incomplete_tasks) == 0:
            print("\n✓ All tasks completed successfully!")
            break

        if not retry_enabled:
            print(
                f"\nRetry disabled. Exiting with {len(incomplete_tasks)} incomplete tasks."
            )
            break

        print(
            f"\nWaiting {args.retry_delay} seconds before retrying {len(incomplete_tasks)} incomplete tasks..."
        )
        time.sleep(args.retry_delay)

        # Update task list to only retry incomplete tasks
        all_tasks = incomplete_tasks
        attempt += 1


if __name__ == "__main__":
    main()
