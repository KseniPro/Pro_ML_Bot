from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, accuracy_score, 
                           classification_report, silhouette_score, r2_score)
import pandas as pd
import numpy as np
from io import BytesIO


def parse_params(param_str: str) -> dict:
    """Парсит строку параметров в словарь"""
    params = {}
    for item in param_str.split(','):
        key, value = item.strip().split('=')
        key = key.strip()
        value = value.strip()
        
        if value.lower() == 'true':
            params[key] = True
        elif value.lower() == 'false':
            params[key] = False
        elif value.lower() == 'none':
            params[key] = None
        elif value.replace('.', '', 1).isdigit():
            params[key] = float(value) if '.' in value else int(value)
        else:
            params[key] = value.strip("'\"")
    return params


def create_model_instance(model_name: str, params: dict, task_type: str):
    """Создает экземпляр модели"""
    if 'регрессия' in task_type:
        if model_name == 'Linear Regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        elif model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        elif model_name == 'XGBoost':
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
    elif 'классификация' in task_type:
        if model_name == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        elif model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif model_name == 'SVM':
            from sklearn.svm import SVC
            return SVC(**params)
    elif 'кластеризация' in task_type:
        if model_name == 'K-Means':
            from sklearn.cluster import KMeans
            return KMeans(**params)
    raise ValueError(f"Неизвестная модель: {model_name}")

def calculate_metrics(y_true, y_pred, task_type: str, X=None):
    """Вычисляет метрики модели по типу задачи."""
    metrics = {}

    if 'регрессия' in task_type:
        mse = mean_squared_error(y_true, y_pred)
        metrics.update({
            'MSE':  mse,
            'RMSE': mse**0.5,
            'R2':   r2_score(y_true, y_pred),
        })

    elif 'классификация' in task_type:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        report: dict = classification_report(y_true, y_pred, output_dict=True)
        if 'weighted avg' in report:
            weighted_avg = report['weighted avg']
            if isinstance(weighted_avg, dict):
                metrics.update(weighted_avg)

    elif 'кластеризация' in task_type:
        if X is None:
            raise ValueError("Для силуэта нужен X (матрица признаков).")
        metrics['Silhouette'] = silhouette_score(X, y_pred)
        # Добавляем количество кластеров и количество точек в каждом кластере
        unique_clusters, counts = np.unique(y_pred, return_counts=True)
        metrics['Number_of_Clusters'] = len(unique_clusters)
        for i, count in zip(unique_clusters, counts):
            metrics[f'Cluster_{i}_Size'] = count

    return metrics

