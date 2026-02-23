"""
Модуль для асинхронного взаимодействия с векторной базой данных Qdrant.

Предоставляет удобный интерфейс для:
- Создания, настройки и удаления коллекций
- Сохранения эмбеддингов (поддержка torch.Tensor и np.ndarray, включая GPU)
- Поиска похожих векторов (по одному или batch-запросам)
- Управления состоянием коллекций

Ключевые улучшения (необходимо реализовать):

3. Внедрён механизм чанкинга (chunking) при сохранении больших батчей эмбеддингов —
   теперь можно безопасно работать с тысячами и миллионами векторов без риска OOM
4. Поддержка фильтрации по payload через параметр `query_filter` в поиске



Архитектура модуля обеспечивает:
- Высокую надёжность за счёт повторных попыток и валидации
- Масштабируемость при работе с большими объёмами данных
- Гибкость в интеграции с RAG-системами и MLOps-пайплайнами

Используется асинхронный движок QdrantClient для неблокирующей работы.
Автоматически обрабатывает перемещение тензоров с GPU на CPU и сериализацию в JSON.
"""
from typing import Sequence
import uuid
import logging
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    OptimizersConfigDiff,
    ScalarQuantization,
    HnswConfigDiff,
    CollectionParamsDiff,
    ScalarQuantizationConfig,
    ScalarType,
)
from .vectorize import vectorize_text
from qdrant_client.conversions import common_types as types
import torch as tr
import numpy as np
from .work_data_config import (
    QDRANT_URL,
    QDRANT_KEY,
)
from .models import PlayloadPoint

logger = logging.getLogger(__name__)


class QdrantConnect:
    """Асинхронный клиент для взаимодействия с векторной базой данных Qdrant.

    Обеспечивает удобные методы для:
    - Создания и настройки коллекций
    - Сохранения эмбеддингов (с поддержкой PyTorch и NumPy)
    - Поиска похожих векторов
    - Управления коллекциями (удаление, получение списка)

    Поддерживает работу с GPU-тензорами: автоматически переводит на CPU,
    удаляет граф вычислений и сериализует для передачи в Qdrant.
    """

    __DEFAULT_VECTOR_COLLECTION_CONFIG_MAP = {
        "vector_params": VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
        "optimizers_params": OptimizersConfigDiff(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            default_segment_number=1,
            max_segment_size=50000,
            memmap_threshold=20000,
            indexing_threshold=20000,
            flush_interval_sec=10,
            max_optimization_threads=2,
        ),
        "hnsw_params": HnswConfigDiff(
            m=48,
            ef_construct=200,
            full_scan_threshold=10000,
        ),
        "collection_params": CollectionParamsDiff(
            on_disk_payload=False,
        ),
        "quantization_params": ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
    }
    """ Карта параметров по умолчанию для создания коллекций.

        Содержит оптимальные значения для большинства задач:
        
        - vector_params: 
            - size=384 — стандартный размер у miniLM и подобных моделей
            - distance=COSINE — косинусная метрика, наиболее подходящая для эмбеддингов
        
        - optimizers_params:
            - deleted_threshold=0.2 — удалять сегменты, если более 20% векторов удалено
            - vacuum_min_vector_number=1000 — не запускать очистку при малом количестве векторов
            - indexing_threshold=20000 — начинать индексацию после 20к векторов
            - flush_interval_sec=10 — регулярная запись на диск
        
        - hnsw_params:
            - m=48 — максимальное количество связей между вершинами
            - ef_construct=200 — качество построения графа
            - full_scan_threshold=10000 — переключение на полный поиск при малых объёмах
        
        - collection_params:
            - on_disk_payload=False — payload хранится в RAM для быстрого доступа
        
        - quantization_params:
            - INT8 + quantile=0.99 — сжатие векторов с минимальной потерей точности
            - always_ram=True — квантованные векторы всегда в оперативной памяти
    """

    def __init__(self, url: str | Path, api_key: str):
        """
        Инициализирует клиент Qdrant.

        :param url: URL-адрес сервера Qdrant.
        :type url: str
        :param api_key: Ключ API для аутентификации.
        :type api_key: str
        """
        self.client = AsyncQdrantClient(url=str(url), api_key=api_key)

    @classmethod
    def _prepare_config_param(
        cls,
        vector_size: int,
        vector_params: VectorParams | None = None,
        optimizers_params: OptimizersConfigDiff | None = None,
        quantization_params: ScalarQuantization | None = None,
        hnsw_params: HnswConfigDiff | None = None,
        collection_params: CollectionParamsDiff | None = None,
    ):
        """
        Формирует словарь параметров конфигурации для создания или обновления коллекции.

        Если параметр не указан, используется значение по умолчанию из __DEFAULT_VECTOR_COLLECTION_CONFIG_MAP.

        :param vector_size: Размерность векторов (например, 384, 768).
        :type vector_size: int
        :param vector_params: Параметры векторов (размер, метрика). Если None — используются значения по умолчанию.
        :type vector_params: VectorParams | None
        :param optimizers_params: Параметры оптимизаторов индексации. Если None — используются значения по умолчанию.
        :type optimizers_params: OptimizersConfigDiff | None
        :param quantization_params: Параметры квантования векторов. Если None — используются значения по умолчанию.
        :type quantization_params: ScalarQuantization | None
        :param hnsw_params: Параметры HNSW-индекса. Если None — используются значения по умолчанию.
        :type hnsw_params: HnswConfigDiff | None
        :param collection_params: Параметры коллекции. Если None — используются значения по умолчанию.
        :type collection_params: CollectionParamsDiff | None

        :return: Словарь с ключами, совместимыми с API Qdrant: vectors_config, optimizers_config и др.
        :rtype: dict

        .. example::
            >>> params = cls._prepare_config_param(768, vector_params=VectorParams(size=768, distance=Distance.DOT))
            >>> # vectors_config будет содержать size=768, distance=DOT
        """
        kwargs = {}

        # Настройка параметров векторов
        # Размер берётся из входного параметра, метрика — из переданного объекта или по умолчанию
        vec_cfg: VectorParams = (
            vector_params or cls.__DEFAULT_VECTOR_COLLECTION_CONFIG_MAP["vector_params"]
        )  # ty:ignore[invalid-assignment] // в любом случае VectorParams если атрибуты класса не изменены
        kwargs["vectors_config"] = VectorParams(
            size=vector_size, distance=vec_cfg.distance
        )

        # Сопоставление внешних параметров с внутренними ключами Qdrant
        # Каждый параметр либо берётся из аргумента, либо из настроек по умолчанию
        mapping = {
            optimizers_params: ("optimizers_config", "optimizers_params"),
            hnsw_params: ("hnsw_config", "hnsw_params"),
            collection_params: ("collection_params", "collection_params"),
            quantization_params: ("quantization_config", "quantization_params"),
        }

        for value, (key, default_key) in mapping.items():
            kwargs[key] = (
                value or cls.__DEFAULT_VECTOR_COLLECTION_CONFIG_MAP[default_key]
            )

        return kwargs

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        vector_params: VectorParams | None = None,
        optimizers_params: OptimizersConfigDiff | None = None,
        quantization_params: ScalarQuantization | None = None,
        hnsw_params: HnswConfigDiff | None = None,
        collection_params: CollectionParamsDiff | None = None,
    ) -> bool:
        """
        Создаёт новую коллекцию в базе данных Qdrant с заданными параметрами.

        Коллекция — это контейнер для хранения векторов и их метаданных (payload).
        После создания можно добавлять в неё точки (векторы).

        :param collection_name: Уникальное имя коллекции. Должно быть строкой без пробелов.
        :type collection_name: str
        :param vector_size: Размерность векторов, которые будут храниться в коллекции.
        :type vector_size: int
        :param vector_params: Пользовательские параметры векторов. Опционально.
        :type vector_params: VectorParams | None
        :param optimizers_params: Параметры оптимизации индексации. Опционально.
        :type optimizers_params: OptimizersConfigDiff | None
        :param quantization_params: Параметры сжатия векторов. Опционально.
        :type quantization_params: ScalarQuantization | None
        :param hnsw_params: Параметры HNSW-графа для поиска. Опционально.
        :type hnsw_params: HnswConfigDiff | None
        :param collection_params: Дополнительные параметры коллекции. Опционально.
        :type collection_params: CollectionParamsDiff | None

        :return: True, если коллекция создана успешно.
        :rtype: bool

        :raises Exception: При ошибке соединения, некорректных параметрах или существующем имени.

        .. note::
            Использует `_prepare_config_param` для формирования полной конфигурации.
            Логгирует успешное создание или ошибку.
        """
        other_param: dict = self._prepare_config_param(
            vector_size,
            vector_params,
            optimizers_params,
            quantization_params,
            hnsw_params,
            collection_params,
        )
        try:
            await self.client.create_collection(
                collection_name=collection_name, **other_param
            )

            logger.info("Коллекция %s успешно создана", collection_name)
            return True

        except Exception as e:
            logger.warning("Ошибка при создании коллекции %s: %s", collection_name, e)
            raise

    async def update_collection_params(
        self,
        collection_name: str,
        vector_size: int,
        vector_params: VectorParams | None = None,
        optimizers_params: OptimizersConfigDiff | None = None,
        quantization_params: ScalarQuantization | None = None,
        hnsw_params: HnswConfigDiff | None = None,
        collection_params: CollectionParamsDiff | None = None,
    ) -> bool:
        """
        Обновляет параметры существующей коллекции.

        Полезно для тонкой настройки производительности после загрузки данных.
        Например, можно изменить пороги оптимизации или включить квантование.

        :param collection_name: Имя существующей коллекции.
        :type collection_name: str
        :param vector_size: Новая размерность векторов (если меняется).
        :type vector_size: int
        :param vector_params: Новые параметры векторов. Опционально.
        :type vector_params: VectorParams | None
        :param optimizers_params: Новые параметры оптимизаторов. Опционально.
        :type optimizers_params: OptimizersConfigDiff | None
        :param quantization_params: Новые параметры квантования. Опционально.
        :type quantization_params: ScalarQuantization | None
        :param hnsw_params: Новые параметры HNSW. Опционально.
        :type hnsw_params: HnswConfigDiff | None
        :param collection_params: Новые параметры коллекции. Опционально.
        :type collection_params: CollectionParamsDiff | None

        :return: True при успешном обновлении.
        :rtype: bool

        :raises Exception: Если коллекция не существует или параметры некорректны.

        .. warning::
            Не все параметры можно изменить после создания. Например, размер вектора.
            Изменения могут потребовать перестроения индекса.
        """
        other_param: dict = self._prepare_config_param(
            vector_size,
            vector_params,
            optimizers_params,
            quantization_params,
            hnsw_params,
            collection_params,
        )

        try:
            await self.client.update_collection(
                collection_name=collection_name, **other_param
            )
            logger.info("Параметры коллекции %s успешно обновлены", collection_name)
            return True
        except Exception as e:
            logger.warning("Ошибка при обновлении коллекции %s: %s", collection_name, e)
            raise

    def _transform_query_to_valid_array(self, query: str | tr.Tensor | np.ndarray) -> list[float]:
        
        if isinstance(query, str):
            embedd = vectorize_text(query)
            
            # Находим вектор средних значений
            query = embedd.mean(dim=0)
            
        #  Преобразуем Tensor в numpy масив
        if isinstance(query, tr.Tensor):
            if query.requires_grad:
                query = query.detach()
            query = query.cpu().numpy()


        # Обработка NumPy массива
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                vector = query
            elif query.ndim == 2:
                # Много векторов — усредняем по первому измерению
                vector = query.mean(axis=0)
            else:
                logger.warning("Ожидался 1D или 2D массив, получено %s D", {query.ndim})
                raise ValueError(f"Ожидался 1D или 2D массив, получено {query.ndim}D")
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            logger.warning("Нулевой вектор — невозможно нормализовать")
            raise ValueError("Нулевой вектор — невозможно нормализовать")

        # Нормализуем вектор 
        vector = vector / norm
        
        return vector.tolist()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, UnexpectedResponse)
        ),
        reraise=True,
    )
    async def find_similar(
        self,
        collection_name: str,
        query: str | tr.Tensor | np.ndarray,
        limit: int = 5,
        query_filter: types.Filter | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float | None = None,
    ) -> list[types.ScoredPoint] | None:
        """
        Найти похожие точки по текстовому запросу.

        Args:
            client: Асинхронный клиент Qdrant
            collection_name: Имя коллекции
            query_text: Текст запроса (будет преобразован в вектор) или вектор
            limit: Количество возвращаемых результатов
            query_filter: Фильтр для ограничения поиска
            with_payload: Включить payload в результат
            with_vectors: Включить векторы в результат
            score_threshold: Минимальный порог схожести

        Returns:
            Список найденных точек с оценками (ScoredPoint)
        """

        embedding = self._transform_query_to_valid_array(query)
        
        response = await self.client.query_points(
            collection_name=collection_name,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )
        return response.points
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, UnexpectedResponse)
        ),
        reraise=True,
    )
    async def find_similar_batch(
        self,
        collection_name: str,
        queries: Sequence[str] | Sequence[tr.Tensor] | Sequence[np.ndarray],
        limit: int = 5,
        query_filter: types.Filter | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float | None= None,
    ) ->  list[list[types.ScoredPoint]]:
        """
        Пакетный поиск похожих точек по нескольким текстовым запросам.

        Args:
            client: Асинхронный клиент Qdrant
            collection_name: Имя коллекции
            queries: Список текстовых запросов
            limit: Количество возвращаемых результатов на каждый запрос
            query_filter: Фильтр для всех запросов
            with_payload: Включить payload в результат
            with_vectors: Включить векторы в результат
            score_threshold: Минимальный порог схожести

        Returns:
            Список списков найденных точек (по одному списку на каждый запрос)
        """
        requests = [
            types.QueryRequest(
                query=self._transform_query_to_valid_array(this_query),
                limit=limit,
                filter=query_filter,  
                with_payload=with_payload,
                with_vector=with_vectors,
                score_threshold=score_threshold,
            )
            for this_query in queries
        ]

        responses = await self.client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )

        # Извлекаем `.points` из каждого ответа
        return [response.points for response in responses]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, UnexpectedResponse)
        ),
        reraise=True,
    )
    async def save_embeddings(
        self,
        embeddings: tr.Tensor | np.ndarray,
        collection_name: str,
        payload_list: list[dict] | None  = None,
        chunk_size: int = 100,
    ) -> bool:
        """
        Сохраняет батч эмбеддингов в указанную коллекцию Qdrant с чанкингом и retry-логикой.

        Принимает как тензоры PyTorch (на GPU/CPU), так и массивы NumPy.
        Автоматически обрабатывает тип данных и перемещает с GPU на CPU.

        :param embeddings: Массив эмбеддингов формы (N, D), где N — количество векторов, D — размерность.
        :type embeddings: torch.Tensor | np.ndarray
        :param collection_name: Имя коллекции, куда сохранять векторы.
        :type collection_name: str
        :param payload_list: Список словарей с метаданными для каждого вектора. Если None — создаются пустые.
        :type payload_list: list[dict] | None
        :param chunk_size: Размер чанка для вставки, по умолчанию 100.
        :type chunk_size: int

        :return: True при успешной вставке всех векторов.
        :rtype: bool

        :raises ValueError: Если форма данных неверна или payload_list не соответствует количеству векторов.
        :raises Exception: При ошибках сети или в Qdrant.

        .. note::
            - Использует `upsert` с `wait=True` — гарантирует запись до возврата.
            - Генерирует уникальные ID с помощью `uuid4()`.
            - Каждый вектор сохраняется как PointStruct с id, vector, payload.
            - Применяется чанкинг для больших батчей.
            - Реализована retry-логика при сетевых ошибках и таймаутах.
        """
        # Преобразование PyTorch тензора в NumPy массив
        if isinstance(embeddings, tr.Tensor):
            # Убираем граф вычислений и перемещаем на CPU
            embeddings = embeddings.detach().cpu().numpy()

        # Валидация входных данных
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("Embeddings должны быть двумерным массивом формы (N, D)")

        num_vectors = embeddings.shape[0]
        payload_list = payload_list or [{} for _ in range(num_vectors)]

        if len(payload_list) != num_vectors:
            raise ValueError(
                "Количество элементов в payload_list должно равняться количеству векторов"
            )

        # Разделение на чанки
        for i in range(0, num_vectors, chunk_size):
            chunk_embeddings = embeddings[i : i + chunk_size]
            chunk_payloads = payload_list[i : i + chunk_size]

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),  # преобразуем в список чисел
                    payload=payload,
                )
                for embedding, payload in zip(chunk_embeddings, chunk_payloads)
            ]

            try:
                # Выполнение асинхронной вставки
                await self.client.upsert(
                    collection_name=collection_name, wait=True, points=points
                )
                logger.info(
                    "Сохранено %d векторов в коллекцию %s", len(points), collection_name
                )
            except Exception as e:
                logger.error(
                    "Ошибка при сохранении векторов в коллекцию %s: %s",
                    collection_name,
                    e,
                )
                raise
        return True

    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        Удаляет коллекцию и все её данные из базы Qdrant.

        Операция необратима. Все векторы и метаданные будут потеряны.

        :param collection_name: Имя коллекции, которую нужно удалить.
        :type collection_name: str

        :raises Exception: Если коллекция не существует или произошла ошибка сети.

        .. warning::
            Убедитесь, что коллекция больше не нужна. Восстановление невозможно.
        """
        try:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info("Коллекция %s успешно удалена", collection_name)
        except Exception as e:
            logger.error("Ошибка при удалении коллекции %s: %s", collection_name, e)
            raise

    async def get_all_collections(self) -> list:
        """
        Получает список имён всех коллекций, доступных в экземпляре Qdrant.

        Полезно для мониторинга, отладки или выбора коллекции для работы.

        :return: Список строк — имён существующих коллекций.
        :rtype: list[str]

        :raises Exception: При ошибке соединения с Qdrant.

        .. example::
            >>> client = get_qdrant_client()
            >>> collections = await client.get_all_colections()
            >>> print(collections)
            ['documents_2024', 'products', 'users']
        """
        try:
            collections = await self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error("Ошибка при получении списка коллекций: %s", e)
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Проверяет, существует ли коллекция в базе данных Qdrant.

        :param collection_name: Имя коллекции для проверки.
        :type collection_name: str
        :return: True, если коллекция существует; иначе False.
        :rtype: bool

        .. example::
            >>> client = get_qdrant_client()
            >>> exists = await client.collection_exists("my_docs")
            >>> if not exists:
            ...     await client.create_collection("my_docs", vector_size=384)
        """
        try:
            await self.client.get_collection(collection_name)
            return True
        except Exception:
            return False


def get_qdrant_client(
    url: str | Path = QDRANT_URL, key: str = QDRANT_KEY
) -> QdrantConnect:
    """
    Фабричная функция для создания экземпляра клиента Qdrant.

    Использует значения по умолчанию из конфигурации проекта, если параметры не заданы.

    :param url: URL-адрес сервера Qdrant. По умолчанию — значение из work_data_config.QDRANT_URL.
    :type url: str | Path
    :param key: Ключ API для аутентификации. По умолчанию — значение из work_data_config.QDRANT_KEY.
    :type key: str

    :return: Готовый к использованию экземпляр QdrantConnect.
    :rtype: QdrantConnect

    .. example::
        >>> client = get_qdrant_client()
        >>> await client.create_collection("my_docs", vector_size=384)
    """
    return QdrantConnect(url, key)
