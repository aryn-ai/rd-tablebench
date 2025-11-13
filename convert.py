import numpy as np
import numpy.typing as npt
from lxml import etree


def html_to_numpy(html_string: str) -> npt.NDArray[np.str_]:
    # Handle empty or invalid HTML
    if not html_string or not html_string.strip():
        return np.array([]).reshape(0, 0).astype(np.str_)

    dom_tree = etree.HTML(html_string, parser=etree.HTMLParser())

    # Handle case where parsing failed
    if dom_tree is None:
        return np.array([]).reshape(0, 0).astype(np.str_)

    table_rows: list[list[str]] = []
    span_info: dict[int, tuple[str, int]] = {}

    for table_row in dom_tree.xpath("//tr"):
        current_row: list[str] = []
        column_index = 0

        while span_info.get(column_index, (None, 0))[1] > 0:
            current_row.append(span_info[column_index][0])
            span_info[column_index] = (
                span_info[column_index][0],
                span_info[column_index][1] - 1,
            )
            if span_info[column_index][1] == 0:
                del span_info[column_index]
            column_index += 1

        for table_cell in table_row.xpath("td|th"):
            while span_info.get(column_index, (None, 0))[1] > 0:
                current_row.append(span_info[column_index][0])
                span_info[column_index] = (
                    span_info[column_index][0],
                    span_info[column_index][1] - 1,
                )
                if span_info[column_index][1] == 0:
                    del span_info[column_index]
                column_index += 1

            row_span = int(table_cell.get("rowspan", "1"))
            col_span = int(table_cell.get("colspan", "1"))
            cell_text = "".join(table_cell.itertext()).strip()

            if row_span > 1:
                for i in range(col_span):
                    span_info[column_index + i] = (cell_text, row_span - 1)

            for _ in range(col_span):
                current_row.append(cell_text)
            column_index += col_span

        while span_info.get(column_index, (None, 0))[1] > 0:
            current_row.append(span_info[column_index][0])
            span_info[column_index] = (
                span_info[column_index][0],
                span_info[column_index][1] - 1,
            )
            if span_info[column_index][1] == 0:
                del span_info[column_index]
            column_index += 1

        table_rows.append(current_row)

    max_columns = max(map(len, table_rows)) if table_rows else 0
    for row in table_rows:
        row.extend([""] * (max_columns - len(row)))

    return np.array(table_rows)
