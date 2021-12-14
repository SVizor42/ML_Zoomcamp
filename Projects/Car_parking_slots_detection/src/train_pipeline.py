import logging
import sys
import click

from src.entities.train_pipeline_params import read_training_pipeline_params
from src.data.process_data import make_dataset
from src.models.model_fit_predict import train_model, get_classification_report

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command(name="train_pipeline")
@click.option("--config_path", default="configs/train_config.yaml")
def train_pipeline(config_path: str):
    """
    The main pipeline for data preprocessing, training and evaluating model.
    :param config_path: training parameters
    :return: nothing
    """
    params = read_training_pipeline_params(config_path)
    train_dir = f'{params.data_dir}/train'
    val_dir = f'{params.data_dir}/val'
    test_dir = f'{params.data_dir}/test'

    logger.info(f'Starting train pipeline with the following params:\n{params}.')
    logger.info('Reading training dataset...')
    target_size = (params.image_params.width, params.image_params.height)
    train_generator = make_dataset(
        params.train_params.base_model,
        train_dir,
        target_size,
        params.train_params.batch_size,
        use_augmentation=True,
        use_shuffle=True
    )

    logger.info('Reading validation dataset...')
    validation_generator = make_dataset(
        params.train_params.base_model,
        val_dir,
        target_size,
        params.train_params.batch_size,
        use_augmentation=False,
        use_shuffle=False
    )

    input_shape = (
        params.image_params.width,
        params.image_params.height,
        params.image_params.n_channels
    )
    model, history = train_model(
        train_generator,
        validation_generator,
        input_shape,
        params.train_params,
        params.model_dir
    )

    logger.info('Building classification report and confusion matrix for val data...')
    get_classification_report(model, validation_generator)

    logger.info('\nReading test dataset...')
    test_generator = make_dataset(
        params.train_params.base_model,
        test_dir,
        target_size,
        params.train_params.batch_size,
        use_augmentation=False,
        use_shuffle=False
    )

    logger.info('Model evaluation on the test data...')
    model.evaluate(test_generator)
    logger.info('Building classification report and confusion matrix for test data...')
    get_classification_report(model, test_generator)
    logger.info('\nModel training successfully completed.')


if __name__ == "__main__":
    train_pipeline()
