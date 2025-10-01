<#!
.SYNOPSIS
    Полная настройка проекта RAG на Windows Server без GPU.
.DESCRIPTION
    Скрипт выполняет установку зависимостей, создаёт виртуальное окружение,
    устанавливает llama-cpp-python с CPU-колёсами и скачивает рекомендованные модели.
#>

$ErrorActionPreference = "Stop"

Write-Host "=== RAG Windows Setup ===" -ForegroundColor Cyan

# --- Определяем корневую директорию репозитория ---
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot "venv"
$pythonExe = "python"

# --- Проверяем Python ---
Write-Host "1. Проверяем Python..."
$pythonVersion = & $pythonExe --version
Write-Host "   Найден: $pythonVersion"

# --- Создаём виртуальное окружение ---
if (-Not (Test-Path $venvPath)) {
    Write-Host "2. Создаём виртуальное окружение..."
    & $pythonExe -m venv $venvPath
} else {
    Write-Host "2. Виртуальное окружение уже существует."
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

# --- Обновляем pip и ставим зависимости ---
Write-Host "3. Устанавливаем зависимости проекта..."
& $venvPython -m pip install --upgrade pip
& $venvPip install -r (Join-Path $repoRoot "requirements.txt")

# --- Устанавливаем llama-cpp-python CPU wheel ---
Write-Host "4. Устанавливаем llama-cpp-python (CPU wheel)..."
& $venvPip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# --- Скачиваем модель GGUF через huggingface_hub ---
Write-Host "5. Скачиваем рекомендованную модель (Qwen2.5 7B Q4_K_M)..."
$modelsDir = Join-Path $repoRoot "models"
if (-Not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

& $venvPython -m huggingface_hub download `
    --repo-id TheBloke/Qwen2.5-7B-Instruct-GGUF `
    --include "*q4_k_m.gguf" `
    --local-dir $modelsDir `
    --local-dir-use-symlinks False

Write-Host "6. Настройка завершена." -ForegroundColor Green
Write-Host "Активируйте окружение командой: `n  `"$venvPath\Scripts\activate`""
Write-Host "После активации запустите индексацию: python rag_index.py examples/example_documents.jsonl"
Write-Host "Затем выполните запросы: python rag_query.py"
