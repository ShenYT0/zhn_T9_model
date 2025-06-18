import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pipeline import Pipeline
import asyncio

async def input_predict(text:str) -> list[str]:
    pipe = Pipeline()
    candidats = pipe.predict(text)
    return candidats