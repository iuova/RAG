"""Fix encoding issues in the text file."""
import codecs

# Читаем файл с правильной кодировкой
with open("data/Data_5000 .txt", "r", encoding="cp1251") as f:
    content = f.read()

# Сохраняем в UTF-8
with open("data/Data_5000_fixed.txt", "w", encoding="utf-8") as f:
    f.write(content)

print("Файл пересохранен в UTF-8")
