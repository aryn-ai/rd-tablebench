import sycamore
from sycamore.transforms.partition import ArynPartitioner
from sycamore.data.document import Document
from functools import partial
from datetime import datetime
from pathlib import Path


input_dir = Path("data/rd-tablebench/pdfs")
output_dir = Path("data/rd-tablebench/providers")



def write_table_html(root_dir: Path, doc: Document) -> Document:
    orig_fname = Path(doc.properties['path']).name
    new_fname = orig_fname.replace(".pdf", ".html")
    out_f = root_dir / new_fname
    with open(out_f, "w") as f:
        try:
            html = doc['table'].to_html()
            f.write(html)
        except Exception:
            f.write('<table></table>')
    return doc


def main():
    output_root = output_dir / datetime.now().isoformat()
    output_root.mkdir()
    print(f"Outputting to {str(output_root)}")
    ctx = sycamore.init()

    ds = ctx.read.binary(paths = str(input_dir), binary_format = "pdf")
    ds = ds.partition(ArynPartitioner(extract_table_structure=True))
    ds = ds.spread_properties(['path']).explode()
    ds = ds.filter(lambda d: d.type == "table")
    ds = ds.map(partial(write_table_html, output_root))
    ds.execute()


if __name__ == "__main__":
    main()
