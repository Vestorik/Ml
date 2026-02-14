import logging
from pathlib import Path
from concurrent_log_handler import ConcurrentRotatingFileHandler
import torch as tr
from dotenv import load_dotenv
from os import environ as env, getenv

load_dotenv()


#  Основные пути
BASE_PATH: Path = Path(__file__).resolve().parent
DATA_STORAGE_PATH = BASE_PATH / "data_storage"
DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_PATH / "logging"
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR /"app.log"


#  Модели
DEVICE: str = tr.device("cuda" if tr.cuda.is_available() else "cpu")
SENTECE_TRANSFORM_MODEL_STR: str = getenv('SENTECE_TRANSFORM_MODEL_NAME', 'sberbank-ai/sbert_large_nlu_ru')


# Qdrant API
VECTOR_BD_URL = getenv("QDRANT_HOST", "http://localhost:") + getenv("QDRANT_PORT", "6333")
VECTOR_BD_API_KEY = getenv("QDRANT_API_KEY", "")


#  Сообщество
env["HUGGINGFACEHUB_API_TOKEN"] = getenv("HUGGING_FASE_TOKEN", "")

# Логирование
loghandler: ConcurrentRotatingFileHandler = ConcurrentRotatingFileHandler(
    filename=str(log_path),
    maxBytes=10 * 1024 * 1024,  # 10 МБ
    backupCount=5,
    encoding="utf-8",
)

loghandler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s - %(filename)s - %(funcName)s - %(lineno)d"
    )
)

# Настраиваем корневой логгер
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(loghandler)