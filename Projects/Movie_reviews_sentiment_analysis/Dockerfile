FROM python:3.8.12-slim

WORKDIR /usr/app
COPY requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet')" ]
RUN cp -r /root/nltk_data /usr/local/share/nltk_data

COPY data/external ./data/external
COPY models/model.pkl ./models/model.pkl
COPY src/entities ./src/entities
COPY src/data ./src/data
COPY predict.py ./

EXPOSE 8000
CMD uvicorn predict:app --host=0.0.0.0 --port=8000
#CMD uvicorn predict:app --host=0.0.0.0 --port=$PORT
