import logging
import io
import os
import sys
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional
from PIL import Image
# for local usage use the string below
import tensorflow.lite as tflite
# in case of deployment use the string below and comment the string above
# import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

from src.entities.parking_slot_response import ParkingSlotResponse

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


interpreter: Optional[tflite.Interpreter] = None

app = FastAPI(title='Parking slots analysis')


@app.on_event("startup")
def load_model():
    global interpreter
    model_path = os.environ['MODEL_PATH']
    if model_path is None:
        err = f'There is no model for the path {model_path}.'
        logger.error(err)
        raise RuntimeError(err)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()


@app.get('/')
def main():
    return 'Welcome to the Car parking slots analysis webservice!'


@app.get('/health')
def health() -> str:
    return f'TFLite model is ready: {interpreter is not None}'


@app.post("/predict", response_model=ParkingSlotResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        logger.info(f'Image file to analyse: {file.filename}')

        # read file and convert it to image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # preprocess image and convert it to tensor
        preprocessor = create_preprocessor('xception', target_size=(150, 150))
        tensor = preprocessor.convert_to_tensor(image)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(input_index, tensor)
        interpreter.invoke()
        probability = interpreter.get_tensor(output_index)[0][0]

        # get slot_id from the image name
        slot_id = os.path.splitext(file.filename)[0].split('_')[-1]
        slot = 'occupied' if (probability >= 0.5) else 'empty'
        logger.info(f'Parking slot #{slot_id} is {slot}.')

        return ParkingSlotResponse(probability=probability, slot=slot)

    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run('app', host='0.0.0.0', port=8000)
