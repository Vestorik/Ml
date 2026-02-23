from typing import AsyncGenerator
from pathlib import Path
import logging
from aiofiles import open as aiopen

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


async def get_text(
    file_path: str | Path, max_text_length: int = 2621440
) -> AsyncGenerator[str]:
    """Асинхронно читает текст из файла построчно и возвращает порции текста заданной длины.

        Функция эффективно обрабатывает большие файлы без полной загрузки в память.
        Пустые строки пропускаются. Слишком длинные строки (длиннее `max_text_length`)
        разбиваются принудительно. Остальные строки объединяются в пакеты до достижения лимита.

        Args:
            file_path (str | Path): Путь к текстовому файлу (кодировка UTF-8).
            max_text_length (int): Максимальная длина одной порции текста в символах.
                По умолчанию 2_621_440 (~10 MiB при 4 байта на символ в UTF-8).

        Yields:
            str: Текстовая порция, готовая к дальнейшей обработке (разбиению или векторизации).

        Example:
            async for chunk in get_text("large_file.txt", max_text_length=1000):
                print(len(chunk))  # <= 1000

        Note:
            - Не включает пустые строки.
            - Гарантирует, что ни одна порция не превысит `max_text_length`.
            - Последняя порция может быть короче.
    """

    # Асинхроно читаем файл
    async with aiopen(file_path, "r", encoding="utf-8") as f:

        sample: list[str] = []
        sample_len = 0

        async for line in f:
            line = line.strip()
            if not line:  # если строка пустая пропускаем
                continue

            len_line = len(line)

            # Одна очень длинная строка — режем её принудительно
            if len_line > max_text_length:
                for i in range(0, len_line, max_text_length):
                    yield line[i : i + max_text_length]
                continue

            # Собираем выборку из строк
            sample.append(line)
            sample_len += len_line

            # Если выборка больше max_text_length, возвращаем её
            if sample_len >= max_text_length:

                text_batch: str = " ".join(sample[:-1])
                sample = sample[-1:]  # Обновляем выборку
                sample_len = len("".join(sample))

                yield text_batch

        # Возвращаем остаток
        if len(sample) > 0:
            yield " ".join(sample)