from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
import torch as tr
import numpy as np

class PlayloadPoint(BaseModel):
    text: str = Field(..., min_length=1, description="Текст чанка")
    source: Optional[str] = Field(None, description="Источник: имя файла, URL и т.п.")
    metadata: Dict[str,  str | int | float] = Field(default_factory=dict, description="Дополнительные поля: title, page и т.п.")
    

