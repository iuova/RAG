# Установка Visual Studio Build Tools для llama-cpp-python

## 🚀 Быстрая установка

### Шаг 1: Скачивание
1. Перейдите на: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Скачайте **"Build Tools for Visual Studio 2022"**
3. Запустите установщик

### Шаг 2: Выбор компонентов
В установщике выберите:
- ✅ **C++ build tools** (основная рабочая нагрузка)
- ✅ **MSVC v143 - VS 2022 C++ x64/x86 build tools**
- ✅ **Windows 10/11 SDK** (последняя версия)
- ✅ **CMake tools for Visual Studio**

### Шаг 3: Установка
1. Нажмите **"Установить"**
2. Дождитесь завершения (10-30 минут)
3. **Перезапустите командную строку**

## 🔧 Проверка установки

### Тест 1: Компилятор
```bash
cl
```
Должно показать информацию о Microsoft C/C++ компиляторе.

### Тест 2: Установка llama-cpp-python
```bash
pip install llama-cpp-python --no-cache-dir
```

### Тест 3: Проверка RAG системы
```bash
python test_llama_cpp.py
```

## 📊 Размер и время

| Компонент | Размер | Время |
|-----------|--------|-------|
| Visual Studio Build Tools | ~3-4 GB | 10-30 мин |
| Windows SDK | ~1-2 GB | - |
| **Общий размер** | **~4-6 GB** | **10-30 мин** |

## 🚨 Устранение проблем

### Проблема 1: "cl не является внутренней или внешней командой"
**Решение:**
1. Перезапустите командную строку
2. Или запустите "Developer Command Prompt for VS"

### Проблема 2: Ошибка компиляции llama-cpp-python
**Решение:**
```bash
# Очистите кэш pip
pip cache purge

# Установите с дополнительными флагами
set CMAKE_ARGS=-DLLAMA_CUBLAS=off
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### Проблема 3: Нехватка места на диске
**Решение:**
- Build Tools требуют ~4-6 GB
- Освободите место на диске C:
- Или установите на другой диск

## 🔄 Альтернативы

Если установка Build Tools вызывает проблемы:

### 1. Transformers (рекомендуется)
```bash
python rag_query_transformers.py
```
- ✅ Работает сразу
- ✅ Не требует компиляции
- ✅ Поддерживает все модели

### 2. Простой поиск
```bash
python rag_query_simple.py
```
- ✅ Быстрый поиск
- ✅ Идеально для тестирования

### 3. Ollama
```bash
# Установка Ollama
# https://ollama.ai/download

# Запуск модели
ollama run qwen2.5:7b
```

## 📋 Чек-лист установки

- [ ] Скачан Visual Studio Build Tools
- [ ] Выбрана рабочая нагрузка "C++ build tools"
- [ ] Включены MSVC v143, Windows SDK, CMake
- [ ] Установка завершена
- [ ] Перезапущена командная строка
- [ ] Проверен компилятор (`cl`)
- [ ] Установлен llama-cpp-python
- [ ] Протестирована RAG система

## 🎯 Результат

После успешной установки:
- ✅ `rag_query_simple.py` - работает
- ✅ `rag_query_transformers.py` - работает  
- ✅ `rag_query.py` - работает (с llama.cpp)

**Система RAG полностью функциональна!**
