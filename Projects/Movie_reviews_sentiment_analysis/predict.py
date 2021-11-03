import logging
import sys
import pickle
import uvicorn
from typing import Optional
from fastapi import FastAPI
from sklearn.pipeline import Pipeline

from src.entities.movie_reviews import MovieReviewData, MovieReviewResponse
from src.data.process_data import process_text

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

MODEL_PATH = 'models/model.pkl'

pipeline: Optional[Pipeline] = None

app = FastAPI(title='Movie reviews sentiment analysis')


def load_object(path: str) -> Pipeline:
    with open(path, 'rb') as f:
        return pickle.load(f)


@app.on_event("startup")
def load_model():
    global pipeline
    if MODEL_PATH is None:
        err = f'There is no model for the path {MODEL_PATH}.'
        logger.error(err)
        raise RuntimeError(err)
    pipeline = load_object(MODEL_PATH)


@app.get('/')
def main():
    return 'Welcome to the Movie reviews sentiment analysis webservice!'


@app.get('/health')
def health() -> str:
    return f'Pipeline is ready: {pipeline is not None}'


@app.post('/predict', response_model=MovieReviewResponse)
def predict(request: MovieReviewData):
    print(f'Incoming request:\n{request}')

    processed_text = process_text(request.review, lemmatize=True)
    probability = pipeline.predict_proba([processed_text])[0, 1]
    sentiment = 'positive' if (probability >= 0.5) else 'negative'
    print(f'Movie review is {sentiment} (probability = {probability:.3f}).')

    return MovieReviewResponse(probability=probability, sentiment=sentiment)


if __name__ == "__main__":
    uvicorn.run('app', host='0.0.0.0', port=8000)
