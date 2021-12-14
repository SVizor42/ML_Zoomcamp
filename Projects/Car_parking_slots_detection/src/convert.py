import os
import click
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


@click.command(name="train_pipeline")
@click.option("--model_path", default="models/model.h5")
def convert(model_path: str):
    """
    A small script for model conversion from h5-format to tflite
    :param model_path: model in old keras format (.h5)
    :return: model in tflite-format
    """
    print(f'Loading model from {model_path} file...')
    model = keras.models.load_model(model_path)

    print(f'Done.\nConverting model to tflite format...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    new_path = os.path.splitext(model_path)[0] + '.tflite'
    print(f'Done.\nSaving model to {new_path} file...')
    with open(new_path, 'wb') as f_out:
        f_out.write(tflite_model)
        print('Done.')


if __name__ == "__main__":
    convert()
