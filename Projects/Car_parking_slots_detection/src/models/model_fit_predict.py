import logging
import os
import sys
import numpy as np
from typing import Tuple, Union
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

from src.entities.train_params import TrainingParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

TFPretrainedModels = Union[Xception, ResNet50, VGG16]


class PrintLRCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f'learning_rate: {K.eval(lr_with_decay):.5f}')


def make_pretrained_model(
        base_model: TFPretrainedModels,
        input_shape: Tuple[int] = (128, 128, 3),
        use_inner: bool = False,
        inner_size: int = 0,
        use_dropout: bool = False,
        drop_rate: float = 0.0,
        loss: str = 'binary_crossentropy',
        optimizer: str = 'adam',
        model_name: str = None
):
    inputs = keras.Input(shape=input_shape)
    base = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(base)

    if use_inner:
        x = layers.Dense(inner_size, activation='relu')(x)

    if use_dropout:
        x = layers.Dropout(drop_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    if model_name is not None:
        model._name = model_name
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def train_model(
        train_generator: tf.data.Dataset,
        validation_generator: tf.data.Dataset,
        input_shape: Tuple,
        train_params: TrainingParams,
        model_dir: str
):
    """
    Trains the model and returns the result.
    :param train_generator: train dataset
    :param validation_generator: validation dataset
    :param input_shape: shape of the input image [width, height, channels]
    :param train_params: training parameters
    :param model_dir: dir to save the models
    :return:
    """
    logger.info('Configuring model architecture...')
    loss = keras.losses.BinaryCrossentropy()

    # pick the right optimizer according to config file
    optimizer_name = str.lower(train_params.optimizer)
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=train_params.learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = keras.optimizers.Nadam(
            learning_rate=train_params.learning_rate, nesterov=True
        )
    else:
        optimizer = keras.optimizers.SGD(
            learning_rate=train_params.learning_rate, momentum=0.8, nesterov=True
        )

    # pick the pretrained model from keras
    model_name = str.lower(train_params.base_model)
    if model_name == 'xception':
        base_model = Xception(
            weights='imagenet', include_top=False, input_shape=input_shape
        )
    elif model_name == 'resnet50':
        base_model = ResNet50(
            weights='imagenet', include_top=False, input_shape=input_shape
        )
    elif model_name == 'vgg16':
        base_model = VGG16(
            weights='imagenet', include_top=False, input_shape=input_shape
        )
    else:
        raise NotImplementedError()
    base_model.trainable = False

    # build a model based on pretrained one
    model_name = f'{train_params.base_model}_model'
    model = make_pretrained_model(
        base_model=base_model,
        input_shape=input_shape,
        use_inner=train_params.use_inner_layer,
        inner_size=train_params.inner_size,
        use_dropout=train_params.use_dropout,
        drop_rate=train_params.drop_rate,
        loss=loss,
        optimizer=optimizer,
        model_name=model_name
    )
    model.summary()

    # initialize callback for making checkpoints
    model_type = ('custom_models' if 'custom' in model._name else 'pretrained_models')
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, model_type, model._name + '_{epoch:02d}_{val_accuracy:.3f}.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    # initialize callback for learning rate scheduler
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001
    )

    # initialize callback to print learning rate
    print_lr = PrintLRCallback()

    logger.info('Model training...')
    history = model.fit(
        train_generator,
        steps_per_epoch=int(len(train_generator) / 2),
        epochs=train_params.epochs,
        validation_data=validation_generator,
        validation_steps=int(len(validation_generator) / 2),
        callbacks=[checkpoint, reduce_lr, print_lr],
    )

    return model, history


def predict_model(model, validation_generator: tf.data.Dataset) -> np.ndarray:
    """
    Makes predictions using model.
    :param model: the pipeline to predict with
    :param validation_generator: validation dataset
    :return: array of hard predictions
    """
    probes = model.predict_generator(validation_generator)
    predicts = probes > 0.5
    return predicts


def get_classification_report(model, validation_generator: tf.data.Dataset):
    y_pred = predict_model(model, validation_generator)
    y_true = validation_generator.classes
    logger.info('Classification report:')
    logger.info(
        classification_report(y_true, y_pred, target_names=['Empty', 'Occupied'])
    )
    logger.info('Confusion matrix:')
    logger.info(confusion_matrix(y_true, y_pred))
