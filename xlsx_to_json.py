import pandas as pd
import json

# === Настройки ===
input_file = "data.xlsx"       # исходный Excel-файл
output_file = "data.jsonl"    # файл для сохранения данных
sheet_name = 0                  # или имя листа, если нужно явно

# === Чтение Excel ===
df = pd.read_excel(input_file, sheet_name=sheet_name)

# Проверим наличие нужных колонок
required_cols = ["Работа", "ПунктРемонтнойВедомости", "Описание"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в файле Excel")

# === Преобразование в JSONL ===
with open(output_file, "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        record = {
            "id": str(idx + 1),
            "text": f"{str(row['Работа']).strip()} {str(row['ПунктРемонтнойВедомости']).strip()} {str(row['Описание']).strip()}",
            "title": str(row["ПунктРемонтнойВедомости"]).strip()
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Файл {output_file} успешно создан с {len(df)} записями.")
