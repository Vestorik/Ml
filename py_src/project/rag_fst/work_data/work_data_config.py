from langsmith.prompt_cache import Cache
from sentence_transformers import SentenceTransformer
from config import DEVICE, SENTECE_TRANSFORM_MODEL_STR, VECTOR_BD_API_KEY, VECTOR_BD_URL, CACHE_PATH
from langchain_text_splitters import RecursiveCharacterTextSplitter

QDRANT_URL: str = VECTOR_BD_URL
QDRANT_KEY: str = VECTOR_BD_API_KEY

# Трансформеры текста
SENTECE_TRANSFORM_MODEL = SentenceTransformer(SENTECE_TRANSFORM_MODEL_STR)
                                            #   cache_folder=str(CACHE_PATH / 'models' / SENTECE_TRANSFORM_MODEL_STR))
SENTECE_TRANSFORM_MODEL = SENTECE_TRANSFORM_MODEL.to(DEVICE)

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
