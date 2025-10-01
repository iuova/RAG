"""Convert Excel documents into the JSONL format expected by rag_index.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DEFAULT_JSONL, ensure_directories


REQUIRED_COLUMNS = ("Работа", "ПунктРемонтнойВедомости", "Описание")


def build_record(idx: int, row: pd.Series) -> dict:
    title = str(row["ПунктРемонтнойВедомости"]).strip()
    text_parts = [str(row[col]).strip() for col in REQUIRED_COLUMNS]
    text = " ".join(part for part in text_parts if part)
    return {"id": str(idx + 1), "title": title, "text": text}


def dataframe_to_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for idx, row in df.iterrows():
            record = build_record(idx, row)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Excel data into JSONL format.")
    parser.add_argument("input", type=Path, help="Path to the source .xlsx file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JSONL,
        help="Destination JSONL file. Defaults to data/data.jsonl.",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet index or name to read (pandas read_excel sheet_name argument).",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel file: {', '.join(missing)}")

    if args.output.is_absolute():
        output_path = args.output
    elif args.output.parent == Path('.'):
        output_path = DATA_DIR / args.output.name
    else:
        output_path = (Path.cwd() / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe_to_jsonl(df, output_path)
    print(f"Файл {output_path} успешно создан с {len(df)} записями.")


if __name__ == "__main__":
    main()
