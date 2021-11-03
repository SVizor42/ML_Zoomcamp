import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def train_model(
    features: pd.DataFrame, target: pd.Series, training_params: dict
) -> Pipeline:
    """
    Trains the pipeline.
    :param features: features to train on
    :param target: target labels to train on
    :param training_params: training parameters
    :return: trained pipeline
    """
    if training_params['transformer_type'] == 'TfidfVectorizer':
        transformer = TfidfVectorizer(
            ngram_range=training_params['transformer_ngram_range'],
            max_df=training_params['transformer_max_df'],
            max_features=training_params['transformer_max_features'],
        )
    else:
        raise NotImplementedError()

    if training_params['model_type'] == 'LogisticRegression':
        model = LogisticRegression(
            C=training_params['model_C'],
            random_state=training_params['model_random_state'],
            max_iter=200,
        )
    else:
        raise NotImplementedError()

    pipeline = Pipeline([('transformer', transformer), ('model', model)])
    pipeline.fit(features, target)

    return pipeline


def predict_model(pipeline: Pipeline, features: pd.DataFrame) -> [np.ndarray, np.ndarray]:
    """
    Makes predictions using pipeline.
    :param pipeline: the pipeline to predict with
    :param features: the features to predict on
    :return: pipeline predictions
    """
    predicts = pipeline.predict(features)
    predict_probes = pipeline.predict_proba(features)
    return [predicts, predict_probes]


def evaluate_model(target: pd.Series, predicts: np.array, predict_probes: np.array) -> dict:
    """
    Evaluates pipeline predictions and returns the metrics.
    :param target: actual target labels
    :param predicts: pipeline hard predictions
    :param predict_probes: pipeline soft predictions
    :return: a dict of metrics in format {'metric_name': value}
    """
    accuracy = round(accuracy_score(target, predicts), 3)
    precision = round(precision_score(target, predicts, average='binary'), 3)
    recall = round(recall_score(target, predicts, average='binary'), 3)
    f1 = round(f1_score(target, predicts, average='binary'), 3)
    roc_auc = round(roc_auc_score(target, predict_probes[:, 1]), 3)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'ROC_AUC': roc_auc
    }
