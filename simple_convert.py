"""Simple conversion script for the data file."""
import json
from pathlib import Path

def convert_data():
    """Convert the data file to JSONL format."""
    
    # Читаем исходный файл
    try:
        with open("data/Data_5000 .txt", "r", encoding="cp1251") as f:
            lines = f.readlines()
    except:
        print("Ошибка чтения файла")
        return
    
    print(f"Прочитано {len(lines)} строк")
    
    # Пропускаем заголовок
    data_lines = lines[1:]
    
    # Создаем JSONL файл
    with open("data/ship_repair_data.jsonl", "w", encoding="utf-8") as outfile:
        for idx, line in enumerate(data_lines):
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            # Очищаем данные
            work_code = parts[0].strip()
            work_item = parts[1].strip() if parts[1] != "<NULL>" else ""
            description = parts[2].strip() if parts[2] not in ["<NULL>", "<Пустая строка>"] else ""
            
            # Пропускаем пустые записи
            if not work_code and not work_item and not description:
                continue
            
            # Формируем текст
            text_parts = []
            if work_code:
                text_parts.append(f"Код работы: {work_code}")
            if work_item:
                text_parts.append(f"Пункт: {work_item}")
            if description:
                text_parts.append(f"Описание: {description}")
            
            text = " | ".join(text_parts)
            title = work_item if work_item else f"Работа {idx + 1}"
            
            record = {
                "id": str(idx + 1),
                "title": title,
                "text": text
            }
            
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print("Конвертация завершена")

if __name__ == "__main__":
    convert_data()
