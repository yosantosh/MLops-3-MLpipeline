import os
import logging
from typing import List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib

# Setup logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Log'))
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "Model_training.log"))
file_handler.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
console_handler.setFormatter(fmt)
file_handler.setFormatter(fmt)

# Avoid adding handlers multiple times if module is reloaded
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Default models to try (classification-focused)
models = [
    LogisticRegression(max_iter=1000),
    # SVC(kernel='rbf', probability=True),
    RandomForestClassifier(n_estimators=130, min_samples_split=5),
    KNeighborsClassifier(n_neighbors=6),
    XGBClassifier(n_estimators=130, use_label_encoder=False, eval_metric='logloss')
]


def load_preprocessed_train(preprocessed_dir: str = None) -> pd.DataFrame:
    """Load the preprocessed train file. By default uses ../Data/preprocessed/train_preprocessed.csv

    Returns:
        DataFrame
    """
    if preprocessed_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        preprocessed_dir = os.path.join(project_root, 'Data', 'preprocessed')

    train_path = os.path.join(preprocessed_dir, 'train_preprocessed.csv')
    if not os.path.exists(train_path):
        logger.error('Train file not found at %s', train_path)
        raise FileNotFoundError(f'Train file not found at {train_path}')

    df = pd.read_csv(train_path)
    logger.info('Loaded preprocessed train data with shape %s', df.shape)
    return df


def split_features_target(df: pd.DataFrame):
    """Split dataframe into X (all columns except last) and y (last column).

    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    if df.shape[1] < 2:
        raise ValueError('Dataframe must have at least one feature column and one target column')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    logger.debug('Split features X shape: %s, target y shape: %s', X.shape, y.shape)
    return X, y


def ensure_models_dir(models_dir: str = None) -> str:
    if models_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'Data', 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def train_and_save(models_list: List, X: pd.DataFrame, y: pd.Series, models_dir: str = None):
    models_dir = ensure_models_dir(models_dir)

    # Decide problem type: classification or regression
    problem_type = 'classification'
    if pd.api.types.is_float_dtype(y.dtype) and y.nunique() > 20:
        problem_type = 'regression'
    elif pd.api.types.is_integer_dtype(y.dtype) and y.nunique() > 20:
        problem_type = 'regression'

    logger.info('Detected problem type: %s', problem_type)

    results = {}
    for model in models_list:
        name = model.__class__.__name__
        try:
            logger.info('Training model: %s', name)
            model.fit(X, y)

            # Evaluate on training data (quick check)
            preds = model.predict(X)
            if problem_type == 'classification':
                report = classification_report(y, preds, zero_division=0)
                logger.info('Classification report for %s:\n%s', name, report)
                score = None
            else:
                mse = mean_squared_error(y, preds)
                r2 = r2_score(y, preds)
                logger.info('%s regression MSE=%.5f R2=%.5f', name, mse, r2)
                score = mse

            # Save model
            model_path = os.path.join(models_dir, f'{name}.pkl')
            joblib.dump(model, model_path)
            logger.info('Saved trained model to %s', model_path)

            results[name] = {
                'model_path': model_path,
                'score': score
            }
        except Exception as e:
            logger.exception('Failed training model %s: %s', name, e)
    return results


def main():
    try:
        df = load_preprocessed_train()
        X, y = split_features_target(df)
        res = train_and_save(models, X, y)
        logger.info('Training completed for %d models', len(res))
    except Exception as e:
        logger.exception('Model training pipeline failed: %s', e)


if __name__ == '__main__':
    main()