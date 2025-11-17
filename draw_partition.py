from aryn_sdk.partition import draw_with_boxes
from pathlib import Path
import json


PROVIDER = "detr-5.0_deformable"
PDF_DIR = Path("data/rd-tablebench/pdfs")
PARTITIONED_DIR = Path("partitioned") / PROVIDER
OUTPUT_DIR = Path("drawn") / PROVIDER

OUTPUT_DIR.mkdir(exist_ok=True)


for partition_json_path in PARTITIONED_DIR.glob("*.json"):
    print(partition_json_path)
    try:
        with open(partition_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {partition_json_path}: {e}")
        continue
    try:
        pdf_file = partition_json_path.stem
        pdf_path = PDF_DIR / pdf_file

        pages = draw_with_boxes(str(pdf_path), data, draw_table_cells=True)

        img_output_file = str(OUTPUT_DIR / pdf_file)
        img_output_file += ".png"
        for i, img in enumerate(pages):
            img.save(img_output_file)

    except Exception as e:
        print(f"Error drawing {partition_json_path}: {e}")
