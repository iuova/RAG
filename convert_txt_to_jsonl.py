"""Convert tab-separated text file into JSONL format for RAG indexing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DATA_DIR, DEFAULT_JSONL, ensure_directories


def convert_txt_to_jsonl(input_file: Path, output_file: Path) -> None:
    """Convert tab-separated text file to JSONL format."""
    
    # Пробуем разные кодировки
    encodings = ['utf-8', 'cp1251', 'latin-1']
    lines = None
    
    for encoding in encodings:
        try:
            with input_file.open("r", encoding=encoding) as infile:
                lines = infile.readlines()
            print(f"Файл успешно прочитан с кодировкой: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if lines is None:
        raise ValueError(f"Не удалось прочитать файл {input_file} ни с одной из кодировок: {encodings}")
    
    # Пропускаем заголовок
    data_lines = lines[1:]
    
    with output_file.open("w", encoding="utf-8") as outfile:
        for idx, line in enumerate(data_lines):
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
                
            # Создаем запись в формате JSONL
            work_code = parts[0].strip() if len(parts) > 0 else ""
            work_item = parts[1].strip() if len(parts) > 1 and parts[1] != "<NULL>" else ""
            description = parts[2].strip() if len(parts) > 2 and parts[2] != "<NULL>" and parts[2] != "<Пустая строка>" else ""
            
            # Формируем заголовок
            title = work_item if work_item else f"Работа {idx + 1}"
            
            # Формируем текст
            text_parts = []
            if work_code:
                text_parts.append(f"Код работы: {work_code}")
            if work_item:
                text_parts.append(f"Пункт ремонтной ведомости: {work_item}")
            if description:
                text_parts.append(f"Описание: {description}")
            
            text = " | ".join(text_parts)
            
            record = {
                "id": str(idx + 1),
                "title": title,
                "text": text
            }
            
            # Убираем пустые значения
            if record["text"].strip():
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert tab-separated text file to JSONL format.")
    parser.add_argument("input", type=Path, help="Path to the source .txt file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JSONL,
        help="Destination JSONL file. Defaults to data/data.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.output.is_absolute():
        output_path = args.output
    elif args.output.parent == Path('.'):
        output_path = DATA_DIR / args.output.name
    else:
        output_path = (Path.cwd() / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_txt_to_jsonl(args.input, output_path)
    print(f"Файл {output_path} успешно создан из {args.input}")


if __name__ == "__main__":
    main()
