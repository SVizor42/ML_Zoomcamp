FROM python:3.9-slim

WORKDIR /usr/app
COPY requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

RUN pip install --no-cache-dir keras-image-helper
RUN pip install --no-cache-dir --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY data/external ./data/external
COPY models/model.tflite ./models/model.tflite
COPY src/entities/parking_slot_response.py ./src/entities/parking_slot_response.py
COPY predict.py ./

ENV MODEL_PATH=models/model.tflite
EXPOSE 8000
CMD uvicorn predict:app --host=0.0.0.0 --port=8000
#CMD uvicorn predict:app --host=0.0.0.0 --port=$PORT