"""
Модуль для асинхронного чтения текстовых файлов и их векторизации с помощью моделей трансформеров.

Данный модуль предоставляет инструменты для эффективной обработки больших текстовых файлов
без полной загрузки в память. Он реализует построчное и порционное чтение, разбиение на чанки
и последующую векторизацию с использованием библиотеки SentenceTransformer.

Основные функции:
- `get_text`: Асинхронно считывает текст из файла, объединяя строки в порции заданной длины.
  Умеет корректно обрабатывать пустые строки и принудительно разбивать слишком длинные строки,
  чтобы избежать переполнения буфера.
- `vectorize_text`: Преобразует текстовые фрагменты в эмбеддинги с помощью предобученной модели.
  Поддерживает пакетную обработку и может работать как на CPU, так и на GPU.

Особенности:
- Полностью асинхронный подход позволяет обрабатывать множество файлов параллельно.
- Использование `RecursiveCharacterTextSplitter` обеспечивает логически целостное разбиение текста.
- Низкое потребление памяти за счёт потоковой обработки.
- Гибкая архитектура: функция `text_func` может быть заменена любой другой стратегией чтения.

Используемые сторонние библиотеки:
- `sentence-transformers`: Для генерации семантических эмбеддингов.
- `langchain-text-splitters`: Для разбиения текста на осмысленные чанки.
- `aiofiles`: Для асинхронного чтения файлов.
- `torch`: Как бэкенд для работы с тензорами.

Типичное использование:
1. Извлечение текста из PDF/документа и сохранение в `.txt`.
2. Постраничная загрузка текста с помощью `get_text`.
3. Векторизация чанков через `vectorize_text` и сохранение в векторную базу данных.

Важно:
- Модуль не включает логику загрузки модели — она должна быть передана извне.
- Все пути могут быть как строками, так и объектами `Path`.
- Кодировка файлов — UTF-8.

"""

from logging import getLogger
from pathlib import Path
from typing import AsyncGenerator, Callable
from sentence_transformers import SentenceTransformer
from aiofiles import open as aiopen
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch as tr


logger = getLogger(__name__)


async def get_text(
    file_path: str | Path, max_text_length: int
) -> AsyncGenerator[str | None]:
    """Модуль для асинхронного чтения текста и его векторизации.

    Содержит функции для постраничного чтения текстовых файлов и преобразования
    текста в эмбеддинги с использованием моделей SentenceTransformer.

    Основные возможности:
    - Асинхронное чтение больших файлов без загрузки в память целиком.
    - Разбиение текста на логические чанки.
    - Пакетная векторизация с использованием GPU/CPU.

    Используемые библиотеки:
    - sentence-transformers
    - langchain-text-splitters
    - torch
    - aiofiles
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


async def vectorize_text(
    model: SentenceTransformer,
    splitter: RecursiveCharacterTextSplitter,
    text_func: Callable[[str | Path, int], AsyncGenerator[str | None, None]],
    file_path: str | Path,
    sample_text_size: int = ((2**20) // 4) * 10,
    vector_batch_size: int = 32,
) -> AsyncGenerator[str | None]:
    """Асинхронно разбивает текст на фрагменты и преобразует их в эмбеддинги с использованием модели трансформера.

    Функция пошагово считывает текст из указанного файла с помощью переданной асинхронной функции,
    разбивает каждый фрагмент на чанки с помощью текстового сплиттера, а затем векторизует эти чанки
    с помощью заданной модели. Результат — поток тензоров с эмбеддингами, возвращаемых по мере готовности.

    Параметр `sample_text_size` определяет приблизительный размер порции текста (в символах), которая
    будет обработана за один шаг. Это позволяет эффективно работать с большими файлами, не загружая
    их целиком в память.

    Args:
        model (SentenceTransformer): Предобученная модель для генерации эмбеддингов.
            Должна поддерживать метод `encode` с возвратом тензоров.
        splitter (RecursiveCharacterTextSplitter): Объект для разбиения текста на логические чанки.
            Обычно используется из библиотеки langchain.
        text_func (Callable[[str | Path, int], AsyncGenerator[str | None, None]]): Асинхронная функция,
            которая принимает путь к файлу и размер выборки, а возвращает генератор строк (фрагментов текста).
            Может возвращать `None`, если фрагмент пуст или недоступен.
        file_path (str | Path): Путь к файлу, содержимое которого необходимо векторизовать.
        sample_text_size (int, optional): Максимальный размер текста (в символах), считываемого за одну итерацию.
            По умолчанию равен приблизительно 10 MiB в пересчёте на UTF-8 (2621440 символов).
            Значение вычисляется как ((2 ** 20) // 4) * 10.
        vector_batch_size (int, optional): Количество текстовых чанков, обрабатываемых одновременно
            при векторизации. Влияет на скорость и потребление памяти. По умолчанию 32.

    Yields:
        torch.Tensor: Двумерный тензор формы (N, D), где N — количество чанков в текущей выборке,
        D — размерность эмбеддингов. Тензор возвращается для каждой обработанной порции текста.

    Example:
        async def read_file_chunks(path: str | Path, size: int) -> AsyncGenerator[str | None, None]:
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                while chunk := await f.read(size):
                    yield chunk

        model = SentenceTransformer('all-MiniLM-L6-v2')
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

        async for embeddings in vectorize_text(model, splitter, read_file_chunks, 'large_document.txt'):
            print(embeddings.shape)  # Например: torch.Size([7, 384])

    Note:
        - Пустые фрагменты текста пропускаются.
        - Модель должна быть совместима с использованием GPU/CPU и поддерживать пакетную обработку.
        - Для работы требуются установленные библиотеки: `sentence-transformers`, `langchain`, `torch`.
    """

    async for sample in text_func(file_path, sample_text_size):
        # Проверяем, что текст не пустой
        if not sample:
            continue

        # Разбиваем текст
        split_sample: list[str] = splitter.split_text(sample)

        # Векторизируем текст
        embedding: tr.Tensor = model.encode(
            split_sample,
            batch_size=vector_batch_size,
            convert_to_tensor=True,
            # show_progress_bar=True, # Показывать прогресс бар
        )

        yield embedding
