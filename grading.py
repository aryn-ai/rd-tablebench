import numpy as np
import numpy.typing as npt
from Levenshtein import distance as levenshtein_distance


BATCH_SIZE = 150

# Scoring parameters (you can adjust these as needed)
S_ROW_MATCH = 5  # Match score for row alignment
G_ROW = -3  # Gap penalty for row alignment (insertion/deletion of a row)
S_CELL_MATCH = 1  # Match score for cell matching
P_CELL_MISMATCH = -1  # Penalty for cell mismatch
G_COL = -1  # Gap penalty for column alignment


def cell_match_score(cell1: str | None, cell2: str | None) -> float:
    """Compute the match score between two cells considering partial matches."""
    if cell1 is None or cell2 is None:
        return P_CELL_MISMATCH  # Penalty for gaps or mismatches
    if cell1 == cell2:
        return S_CELL_MATCH  # Cells are identical

    # Compute the Levenshtein distance using the optimized library
    distance = levenshtein_distance(cell1, cell2)
    max_len = max(len(cell1), len(cell2))
    if max_len == 0:
        normalized_distance = 0.0  # Both cells are empty strings
    else:
        normalized_distance = distance / max_len
    similarity = 1.0 - normalized_distance  # Similarity between 0 and 1
    match_score = P_CELL_MISMATCH + similarity * (S_CELL_MATCH - P_CELL_MISMATCH)
    return match_score


def needleman_wunsch(
    seq1: list[str], seq2: list[str], gap_penalty: int
) -> tuple[list[str | None], list[str | None], float]:
    """
    Perform Needleman-Wunsch alignment between two sequences with free end gaps.

    Parameters:
    seq1, seq2: sequences to align (lists of strings)
    gap_penalty: penalty for gaps (insertions/deletions)

    Returns:
    alignment_a, alignment_b: aligned sequences with gaps represented by None
    score: total alignment score
    """
    m = len(seq1)
    n = len(seq2)

    # Initialize the scoring matrix
    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)
    traceback = np.full((m + 1, n + 1), None)

    # Initialize the first row and column (no gap penalties for leading gaps)
    for i in range(1, m + 1):
        traceback[i, 0] = "up"
    for j in range(1, n + 1):
        traceback[0, j] = "left"

    # Fill the rest of the matrix
    for i in range(1, m + 1):
        seq1_i = seq1[i - 1]
        for j in range(1, n + 1):
            seq2_j = seq2[j - 1]
            match = score_matrix[i - 1, j - 1] + cell_match_score(seq1_i, seq2_j)
            delete = score_matrix[i - 1, j] + gap_penalty
            insert = score_matrix[i, j - 1] + gap_penalty
            max_score = max(match, delete, insert)
            score_matrix[i, j] = max_score
            if max_score == match:
                traceback[i, j] = "diag"
            elif max_score == delete:
                traceback[i, j] = "up"
            else:
                traceback[i, j] = "left"

    # Traceback from the position with the highest score in the last row or column
    i, j = m, n
    max_score = score_matrix[i, j]
    max_i, max_j = i, j
    # Find the maximum score in the last row and column for free end gaps
    last_row = score_matrix[:, n]
    last_col = score_matrix[m, :]
    if last_row.max() > max_score:
        max_i = last_row.argmax()
        max_j = n
        max_score = last_row[max_i]
    if last_col.max() > max_score:
        max_i = m
        max_j = last_col.argmax()
        max_score = last_col[max_j]

    # Traceback to get the aligned sequences
    alignment_a: list[str | None] = []
    alignment_b: list[str | None] = []
    i, j = max_i, max_j
    while i > 0 or j > 0:
        tb_direction = traceback[i, j]
        if i > 0 and j > 0 and tb_direction == "diag":
            alignment_a.insert(0, seq1[i - 1])
            alignment_b.insert(0, seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or tb_direction == "up"):
            alignment_a.insert(0, seq1[i - 1])
            alignment_b.insert(0, None)  # Gap in seq2
            i -= 1
        elif j > 0 and (i == 0 or tb_direction == "left"):
            alignment_a.insert(0, None)  # Gap in seq1
            alignment_b.insert(0, seq2[j - 1])
            j -= 1
        else:
            break  # Should not reach here

    return alignment_a, alignment_b, max_score


def table_similarity(
    ground_truth: npt.NDArray[np.str_], prediction: npt.NDArray[np.str_]
) -> float:
    """
    Compute the similarity between two tables represented as ndarrays of strings,
    allowing for a subset of rows at the top or bottom without penalization (to avoid penalizing subtable cropping).

    Parameters:
    ground_truth, prediction: ndarrays of strings representing the tables

    Returns:
    similarity: similarity score between 0 and 1
    """

    # Handle empty tables
    if ground_truth.size == 0 or prediction.size == 0:
        if ground_truth.size == 0 and prediction.size == 0:
            return 1.0  # Both empty, perfect match
        return 0.0  # One empty, one not - no match

    # Remove newlines and normalize whitespace in cells
    def normalize_cell(cell: str) -> str:
        return "".join(cell.replace("\n", " ").replace("-", "").split()).replace(
            " ", ""
        )

    # Apply normalization to both ground truth and prediction arrays
    vectorized_normalize = np.vectorize(normalize_cell)
    ground_truth = vectorized_normalize(ground_truth)
    prediction = vectorized_normalize(prediction)

    # Convert to lists of lists for easier manipulation
    gt_rows = [list(row) for row in ground_truth]
    pred_rows = [list(row) for row in prediction]

    # Precompute the column alignment scores between all pairs of rows
    m = len(gt_rows)
    n = len(pred_rows)
    row_match_scores = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        gt_row = gt_rows[i]
        for j in range(n):
            pred_row = pred_rows[j]
            # Align columns of the two rows
            _, _, col_score = needleman_wunsch(gt_row, pred_row, G_COL)
            # Adjusted row match score
            row_match_scores[i, j] = col_score + S_ROW_MATCH

    # Initialize the scoring matrix for row alignment with free end gaps
    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)
    traceback = np.full((m + 1, n + 1), None)

    # No gap penalties for leading gaps
    for i in range(1, m + 1):
        traceback[i, 0] = "up"
    for j in range(1, n + 1):
        traceback[0, j] = "left"

    # Fill the rest of the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1, j - 1] + row_match_scores[i - 1, j - 1]
            delete = score_matrix[i - 1, j] + G_ROW
            insert = score_matrix[i, j - 1] + G_ROW
            max_score = max(match, delete, insert)
            score_matrix[i, j] = max_score
            if max_score == match:
                traceback[i, j] = "diag"
            elif max_score == delete:
                traceback[i, j] = "up"
            else:
                traceback[i, j] = "left"

    # Traceback from the position with the highest score in the last row or column
    i, j = m, n
    max_score = score_matrix[i, j]
    max_i, max_j = i, j
    # Find the maximum score in the last row and column for free end gaps
    last_row = score_matrix[:, n]
    last_col = score_matrix[m, :]
    if last_row.max() > max_score:
        max_i = last_row.argmax()
        max_j = n
        max_score = last_row[max_i]
    if last_col.max() > max_score:
        max_i = m
        max_j = last_col.argmax()
        max_score = last_col[max_j]

    # Traceback to get the aligned rows
    alignment_gt_rows: list[list[str | None]] = []
    alignment_pred_rows: list[list[str | None]] = []
    i, j = max_i, max_j
    while i > 0 or j > 0:
        tb_direction = traceback[i, j]
        if i > 0 and j > 0 and tb_direction == "diag":
            alignment_gt_rows.insert(0, gt_rows[i - 1])
            alignment_pred_rows.insert(0, pred_rows[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or tb_direction == "up"):
            alignment_gt_rows.insert(0, gt_rows[i - 1])
            alignment_pred_rows.insert(
                0, [None] * len(gt_rows[i - 1])
            )  # Gap in prediction
            i -= 1
        elif j > 0 and (i == 0 or tb_direction == "left"):
            alignment_gt_rows.insert(
                0, [None] * len(pred_rows[j - 1])
            )  # Gap in ground truth
            alignment_pred_rows.insert(0, pred_rows[j - 1])
            j -= 1
        else:
            break  # Should not reach here

    # Compute the actual total score
    actual_total_score = max_score

    # Compute the total possible score
    num_aligned_rows = len(alignment_gt_rows)
    if num_aligned_rows == 0:
        return 0.0  # Avoid division by zero
    max_row_score = num_aligned_rows * (S_ROW_MATCH + len(gt_rows[0]) * S_CELL_MATCH)
    total_possible_score = max_row_score

    # Normalize the similarity score
    similarity = actual_total_score / total_possible_score
    return float(max(0.0, min(similarity, 1.0)))
