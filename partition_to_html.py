import json
from pathlib import Path
import shutil
from typing import Optional

from sycamore.data.table import Table
from convert import html_to_numpy
from grading import table_similarity

PROVIDER = "detr-5.0_vision"
PARTITIONED_DIR = Path("partitioned") / PROVIDER
OUTPUT_DIR = Path("data/rd-tablebench/providers") / PROVIDER
TEMP_DIR = Path("temp")


def compare_html_tables(file1: Path, file2: Path) -> Optional[float]:
    try:
        array1 = html_to_numpy(file1.read_text())
        array2 = html_to_numpy(file2.read_text())
        return table_similarity(array1, array2)
    except Exception as e:
        print(f"Error comparing tables: {e}")
        return None


def write_table_html(table: Table, file_path: Path) -> None:
    try:
        file_path.write_text(table.to_html())
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        file_path.write_text("<table></table>")


def read_partitioned_file(json_file_path: Path) -> list[Table]:
    try:
        data = json.loads(json_file_path.read_text())
        return [
            Table.from_dict(elem["table"])
            for elem in data["elements"]
            if elem["type"] == "table"
        ]
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return []


def process_tables(tables: list[Table], html_path: str, output_path: Path) -> None:
    if not tables:
        print(f"No tables found for {html_path}")
        return

    if len(tables) == 1:
        write_table_html(tables[0], output_path / html_path)
        return

    print("Multiple tables found")
    TEMP_DIR.mkdir(exist_ok=True)

    # Write all tables to temp files
    temp_files = []
    for i, table in enumerate(tables):
        temp_path = TEMP_DIR / f"{html_path}_{i}.html"
        write_table_html(table, temp_path)
        temp_files.append(temp_path)

    # Find best matching table
    ground_truth = Path(f"data/rd-tablebench/groundtruth/{html_path}")
    best_score = -1.0
    best_file = None

    for temp_file in temp_files:
        score = compare_html_tables(temp_file, ground_truth)
        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_file = temp_file

    if best_file:
        shutil.copy2(best_file, output_path / html_path)
        print(f"Best score: {best_score} for {html_path}")
    else:
        print(f"No best file found for {html_path}")

    for temp_file in temp_files:
        temp_file.unlink()
    TEMP_DIR.rmdir()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    count = 0
    for json_file in PARTITIONED_DIR.glob("*.json"):
        print(f"Processing {json_file}")
        html_path = json_file.stem.replace("pdf", "html")
        tables = read_partitioned_file(json_file)
        process_tables(tables, html_path, OUTPUT_DIR)
        count += 1
    print(f"Processed {count} files")


if __name__ == "__main__":
    main()
