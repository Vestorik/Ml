
import asyncio
import torch as tr
from config import DATA_STORAGE_PATH
from work_data.work_data_config import SENTECE_TRANSFORM_MODEL
from work_data.vectorize import get_text, vectorize_text_file
from work_data.qdrant_connect import get_qdrant_client


test_pdf_path = (
        DATA_STORAGE_PATH / "pdf" / "test" / "Python programming language.pdf"
    )
test_text_path = (
        DATA_STORAGE_PATH
        / "text"
        / "test"
        / "Python programming language.txt"
    )

async def main():
    ...

if __name__ == "__main__":
    asyncio.run(main())




    
    