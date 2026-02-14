from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def check_file(file_path: str | Path) -> bool:
    try:
        # Проверяем, что файл существует
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл {file_path} не найден")

        # Проверяем, что файл не пустой
        if Path(file_path).stat().st_size == 0:
            raise ValueError(f"Файл {file_path} пустой")

    except FileNotFoundError:
        logger.info("Файл %s не найден", file_path)
        return False

    except ValueError:
        logger.info("Файл %s пустой ", file_path)
        return False
    return True
