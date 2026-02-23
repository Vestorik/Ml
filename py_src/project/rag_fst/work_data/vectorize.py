"""
Модуль для асинхронного чтения текстовых файлов и их векторизации с использованием моделей трансформеров.

Данный модуль предназначен для эффективной обработки больших объёмов текста без полной загрузки в память.
Он реализует построчное и порционное чтение файлов, логическое разбиение на чанки и последующую векторизацию
с помощью библиотеки SentenceTransformer. Подходит для интеграции в RAG-системы (Retrieval-Augmented Generation).

Основные функции:
- `vectorize_text`: Преобразует текстовые фрагменты в семантические эмбеддинги с помощью предобученной модели.
  Поддерживает пакетную обработку и может работать как на CPU, так и на GPU.
- `vectorize_text_file`: Комбинирует чтение и векторизацию — позволяет асинхронно обрабатывать файлы
  порциями, что критично при работе с большими документами.

Особенности:
- Полностью асинхронная архитектура: позволяет параллельно обрабатывать множество файлов.
- Низкое потребление памяти за счёт потокового чтения и обработки.
- Использование `RecursiveCharacterTextSplitter` обеспечивает логически целостное разбиение текста
  (например, по абзацам, предложениям).
- Гибкая структура: функцию чтения текста (`text_func`) можно заменить пользовательской реализацией.

Используемые сторонние библиотеки:
- `sentence-transformers`: Для генерации семантических эмбеддингов.
- `langchain-text-splitters`: Для разбиения текста на осмысленные чанки.
- `aiofiles`: Для асинхронного чтения файлов.
- `torch`: Как бэкенд для работы с тензорами.

Типичное использование:
1. Извлечь текст из PDF, DOCX или другого формата и сохранить в `.txt`.
2. Построчно загрузить текст с помощью `get_text` или `vectorize_text_file`.
3. Векторизовать чанки и сохранить в векторную базу данных (например, Chroma, FAISS, Pinecone).

Важно:
- Модуль не включает логику загрузки модели — она должна быть передана извне.
- Все пути могут быть как строками, так и объектами `Path`.
- Файлы должны быть в кодировке UTF-8.
- Рекомендуется проверять существование и непустоту файла до вызова функций (можно использовать `check_file` из `work_data_tool`).
- Функция `get_text` из `work_data_tool` (используется по умолчанию) автоматически пропускает пустые строки и режет слишком длинные строки.
"""

from functools import partial
from logging import getLogger
from pathlib import Path
from typing import AsyncGenerator, Callable
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch as tr
from .work_data_config import SENTECE_TRANSFORM_MODEL, SPLITTER
from .work_data_tool import get_text
from .models import PlayloadPoint

logger = getLogger(__name__)


def vectorize_text(
    text: str,
    splitter: RecursiveCharacterTextSplitter = SPLITTER,
    model: SentenceTransformer = SENTECE_TRANSFORM_MODEL,
    vector_batch_size: int = 32,
    
    # playloads args
    no_playload: bool = True,
    text_source: str = '',
    
    
) -> tuple[tr.Tensor, list[dict] | None]:
    """Преобразует текст в тензор эмбеддингов с помощью модели трансформеров.

    Функция разбивает входной текст на логически связные чанки с использованием переданного
    сплиттера, затем кодирует каждый чанк в векторное представление (эмбеддинг) с помощью
    указанной модели. Результат возвращается в виде тензора PyTorch.

    Аргументы:
        text (str): Текст для векторизации. Должен быть непустой строкой.
        splitter (RecursiveCharacterTextSplitter): Объект для разбиения текста на чанки.
            Определяет размер чанка, перекрытие и символы разбиения.
        model (SentenceTransformer): Модель для генерации эмбеддингов.
            По умолчанию используется глобальная модель из конфигурации.
        vector_batch_size (int): Количество чанков, обрабатываемых за один проход модели.
            Увеличение значения ускоряет обработку, но требует больше памяти. По умолчанию 32.

    Возвращает:
        torch.Tensor: Двумерный тензор формы (N, D), где:
            - N — количество полученных чанков после разбиения,
            - D — размерность эмбеддингов модели (например, 384 для all-MiniLM-L6-v2).

    Пример:
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> model = SentenceTransformer("all-MiniLM-L6-v2")
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        >>> text = "Длинный текст для векторизации..."
        >>> embeddings = vectorize_text(text, splitter, model)
        >>> print(embeddings.shape)
        torch.Size([3, 384])

    Замечания:
        - Модель должна поддерживать вывод эмбеддингов в формате тензора (convert_to_tensor=True).
        - Пустые или слишком короткие тексты могут привести к пустому списку чанков.
        - Для больших текстов рекомендуется использовать построчную загрузку и порционную обработку.
    """
    payloads = None
    
    # Разбиваем текст
    split_sample: list[str] = splitter.split_text(text)

    # Векторизируем текст
    embeddings: tr.Tensor = model.encode(
        split_sample,
        batch_size=vector_batch_size,
        convert_to_tensor=True,
        # show_progress_bar=True,  # Показывать прогресс бар
    )
    if not no_playload:
        payloads: list[dict] = [
            PlayloadPoint(
                text=split_sample[i],
                source=text_source, 
            ).model_dump()
            for i in range(len(split_sample))]
    

    return embeddings, payloads


async def vectorize_text_file(
    file_path: str | Path,
    sample_text_size: int | None = None,
    model: SentenceTransformer | None = None,
    splitter: RecursiveCharacterTextSplitter | None = None,
    text_func: Callable[[str | Path, int], AsyncGenerator[str | None, None]] = get_text,
    vector_batch_size: int | None = None,
) -> AsyncGenerator[tuple[tr.Tensor, list[dict] | None]]:
    """Асинхронно векторизует текст из файла порциями.

    Читает файл фрагментами с помощью `text_func`, разбивает каждый фрагмент на чанки
    и преобразует их в эмбеддинги. Возвращает поток тензоров — по одному на каждую порцию.

    Подходит для обработки очень больших файлов с низким потреблением памяти.

    Args:
        file_path (str | Path): Путь к входному `.txt` файлу.

    Optional args:
        sample_text_size (int): Приблизительный размер текстовой порции в символах.
            По умолчанию 2_621_440 (~10 MiB). Используется как `max_text_length` в `text_func`.
        model (SentenceTransformer | None): Модель для векторизации.
            Если None — используется значение по умолчанию из конфигурации.
        splitter (RecursiveCharacterTextSplitter | None): Сплиттер для чанков.
            Если None — создаётся сплиттер с `chunk_size=500`, `overlap=50`.
        text_func (Callable): Асинхронная функция для чтения текста.
            Должна принимать `(path, max_length)` и возвращать `AsyncGenerator[str]`.
            По умолчанию — `get_text`.
        vector_batch_size (int): Размер пакета при векторизации чанков. По умолчанию 32.

    Yields:
        torch.Tensor: Тензор формы `(N, D)`, где N — число чанков в текущем фрагменте,
        D — размерность эмбеддингов. Каждый тензор соответствует одной порции текста.

    Example:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

        async for embeddings in vectorize_text_file(
            "data/book.txt",
            sample_text_size=1_000_000,
            model=model,
            splitter=splitter
        ):
            print(f"Обработано чанков: {len(embeddings)}")
            # Отправить в векторную БД и т.д.

    Note:
        - Пустые фрагменты текста пропускаются.
        - Для работы требуется установленный `sentence-transformers`, `torch`, `aiofiles`.
        - Файл должен быть в кодировке UTF-8.
        - Не проверяет существование файла — это нужно делать до вызова.
    """
    
    # Параметры для векторизации
    vector_args = {}
    if model is not None:
        vector_args["model"] = model
    if vector_batch_size is not None:
        vector_args["vector_batch_size"] = vector_batch_size
    if splitter is not None:
        vector_args["splitter"] = splitter
        
    if sample_text_size:
        text_func = partial(text_func, max_text_length=sample_text_size)

    async for sample in text_func(file_path):
        # Проверяем, что текст не пустой
        if not sample:
            continue

        data = vectorize_text(sample,no_playload=False, text_source=str(file_path), **vector_args)

        yield data
