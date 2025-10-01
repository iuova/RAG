# === 1. Создание рабочей папки ===
$baseDir = "C:\Users\O.Iunina\Desktop\Projects\RAG"
$modelDir = "$baseDir\models"
if (!(Test-Path $modelDir)) { New-Item -ItemType Directory -Force -Path $modelDir | Out-Null }

# === 2. Создание и активация виртуального окружения ===
Set-Location $baseDir
python -m venv venv
.\venv\Scripts\activate

# === 3. Установка llama-cpp-python через wheel для CPU ===
pip install --upgrade pip
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# === 4. Установка huggingface_hub ===
pip install huggingface_hub

# === 5. Python скрипт для загрузки модели и теста ===
$pythonScript = @"
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

# путь для сохранения модели
model_dir = r"$modelDir"

# скачивание модели (существующая версия Qwen-14B 4bit GGUF)
model_file = hf_hub_download(
    repo_id="TheBloke/Qwen-14B-4bit-GGUF",
    filename="qwen-14b.Q4_K_M.gguf",
    cache_dir=model_dir
)

# инициализация модели
llm = Llama(model_path=model_file, n_threads=32)

# тестовый запрос
prompt = "Объясни, что такое Windows Server 2019, коротко и по делу."
response = llm(prompt, max_tokens=256)
print(response['choices'][0]['text'])
"@

$scriptFile = "$baseDir\test_qwen.py"
$pythonScript | Out-File -FilePath $scriptFile -Encoding UTF8

# === 6. Запуск теста ===
python $scriptFile
