"""Централизованная настройка логирования для RAG проекта."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from config import LOG_DIR, ensure_directories


def setup_logging(
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Настраивает логирование для всего проекта.

    Args:
        log_file: Путь к файлу лога. Если None, используется LOG_DIR / "rag.log"
        level: Уровень логирования (по умолчанию: INFO)
        format_string: Формат строки лога. Если None, используется стандартный формат
        console_output: Выводить ли логи в консоль

    Returns:
        Настроенный логгер
    """
    ensure_directories()

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file is None:
        log_file = LOG_DIR / "rag.log"
    elif isinstance(log_file, str):
        log_file = Path(log_file)

    # Создаем директорию для логов, если её нет
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8")
    ]

    if console_output:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено. Файл лога: {log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Получает логгер с указанным именем.

    Args:
        name: Имя логгера (обычно __name__ модуля)

    Returns:
        Логгер
    """
    return logging.getLogger(name)

