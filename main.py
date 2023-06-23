from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class TextData(BaseModel):
    text: str

with open("keybert.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/")
async def get_keywords_from_text(text_data: TextData):
    keywords = model.extract_keywords(text_data.text, top_n=10, stop_words="english")
    filtered_keywords = [
        [keyword, distance] for keyword, distance in keywords if distance >= 0
    ]
    return dict(filtered_keywords)
