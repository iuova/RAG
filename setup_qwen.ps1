# === 1. Проверка Python ===
Write-Host "Проверка версии Python..."
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python не установлен. Установите Python 3.10+ вручную." -ForegroundColor Red
    exit 1
}

# === 2. Создание рабочей папки ===
$baseDir = "C:\Users\O.Iunina\Desktop\Projects\RAG"
$modelDir = "$baseDir\models"
if (!(Test-Path $modelDir)) {
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
}

# === 3. Создание и активация виртуального окружения ===
Write-Host "Создаём виртуальное окружение..."
Set-Location $baseDir
python -m venv venv
.\venv\Scripts\activate

# === 4. Установка llama-cpp-python (готовый wheel для CPU) ===
Write-Host "Устанавливаем llama-cpp-python..."
pip install --upgrade pip
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# === 5. Установка HuggingFace CLI для загрузки моделей ===
Write-Host "Устанавливаем huggingface_hub..."
pip install huggingface_hub

# === 6. Скачивание модели Qwen-14B Q4_K_M (GGUF) ===
Write-Host "Скачиваем модель Qwen-14B Q4_K_M..."
$repo = "TheBloke/Qwen-14B-GGUF"
$file = "qwen-14b.Q4_K_M.gguf"
huggingface-cli download $repo $file --local-dir $modelDir

# === 7. Создаём тестовый Python-скрипт ===
$testScript = @"
from llama_cpp import Llama

model_path = r"$modelDir\$file"
llm = Llama(model_path=model_path, n_threads=32)  # можно увеличить до 64 при 4 CPU

prompt = "Объясни, что такое Windows Server 2019, коротко и по делу."
response = llm(prompt, max_tokens=256)
print(response["choices"][0]["text"])
"@

$testFile = "$baseDir\test_qwen.py"
$testScript | Out-File -FilePath $testFile -Encoding UTF8

# === 8. Запускаем тест ===
Write-Host "Запускаем тестовую генерацию..."
python $testFile
