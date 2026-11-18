from pathlib import Path
import pandas as pd


def load_excel(file_path: str) -> list[dict]:
    """
    Extract text from an Excel file, one block per sheet.

    Args:
        file_path: path to the .xlsx / .xls file

    Returns:
        List of dicts: [{"text": str, "page": None, "source": str, "sheet": str}, ...]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel not found: {file_path}")

    # Read all sheets
    sheets = pd.read_excel(file_path, sheet_name=None)  # dict: {sheet_name: DataFrame}
    blocks = []

    for sheet_name, df in sheets.items():
        if df.empty:
            continue

        # Drop fully-empty rows/columns, then convert to tab-separated text
        df = df.dropna(how="all").dropna(axis=1, how="all")
        text = df.to_csv(sep="\t", index=False).strip()

        if text:
            blocks.append({
                "text": text,
                "page": None,
                "source": path.name,
                "sheet": sheet_name,
            })

    return blocks