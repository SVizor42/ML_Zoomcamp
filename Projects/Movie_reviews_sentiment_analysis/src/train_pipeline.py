import json
import logging
import sys
import pickle
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data.process_data import read_config, parse_config, process_text
from models.model_fit_predict import train_model, predict_model, evaluate_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command(name="train_pipeline")
@click.option("--config_path", default="configs/train_config.yaml")
def train_pipeline(config_path: str):
    """
    The pipeline to prepare and transform data, train and evaluate model and store the artifacts.
    :param config_path: training parameters
    :return: nothing
    """
    config = read_config(config_path)
    training_params = parse_config(config)

    logger.info(f'Starting train pipeline with the following params: {training_params}.')
    logger.info('Reading data...')
    df = pd.read_csv(training_params['input_data_path'], compression='zip')
    logger.info(f'Initial dataset shape: {df.shape}.')

    logger.info('Processing of the initial dataset...')
    df['review'] = df['review'].apply(process_text, lemmatize=True)
    df['sentiment'] = (df['sentiment'] == 'positive').astype(int)
    logger.info('Dataset was successfully processed.')

    logger.info('Splitting dataset into training and validation parts...')
    train_df, test_df = train_test_split(df,
                                         test_size=training_params['split_val_size'],
                                         shuffle=True,
                                         random_state=training_params['split_random_state']
                                         )
    logger.info('Done.')

    logger.info('Dropping duplicates from the training dataset...')
    train_df = train_df.drop_duplicates()
    X_train, y_train = train_df['review'], train_df['sentiment']
    X_test, y_test = test_df['review'], test_df['sentiment']
    logger.info(f'Training dataset shape: {X_train.shape}, validation part shape: {X_test.shape}.')

    logger.info('Training the pipeline...')
    pipeline = train_model(X_train, y_train, training_params)
    model_name = type(pipeline['model']).__name__
    y_pred_train, probes_train = predict_model(pipeline, X_train)
    train_metrics = evaluate_model(y_train, y_pred_train, probes_train)
    logger.info(f'{model_name} metrics on training data: {train_metrics}')

    logger.info('Validating the pipeline...')
    y_pred_test, probes_test = predict_model(pipeline, X_test)
    test_metrics = evaluate_model(y_test, y_pred_test, probes_test)
    logger.info(f'{model_name} metrics on validation data: {test_metrics}')

    logger.info('Classification report and confusion matrix for the validation data:\n')
    report = classification_report(y_test, y_pred_test, target_names=['negative', 'positive'])
    logger.info(report)
    logger.info(confusion_matrix(y_test, y_pred_test))

    logger.info(f'\nSaving metrics to {training_params["metric_path"]} file...')
    with open(training_params['metric_path'], 'w') as metric_file:
        json.dump(test_metrics, metric_file)

    logger.info(f'Saving training pipeline to {training_params["output_model_path"]} file...')
    with open(training_params['output_model_path'], 'wb') as model_file:
        pickle.dump(pipeline, model_file)

    logger.info('Done.')


if __name__ == "__main__":
    train_pipeline()
