from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

import solutions.config as config


class Task2solver:
    def __init__(self):
        self.RANDOM_STATE = config.RANDOM_STATE

    @staticmethod
    def save_dataframe(df, file_path='.', file_name='file.csv'):
        path = Path(file_path).joinpath(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def split_dataset(self, df, test_size=0.2):
        train, test = train_test_split(
            df, test_size=test_size, random_state=self.RANDOM_STATE)
        return train, test

    def prepare_dataset(self, df_path):
        df = pd.read_csv(df_path)
        df = df.drop('Unnamed: 0', axis=1)
        train_test, val = self.split_dataset(df, test_size=0.1)
        self.save_dataframe(val, file_path='../processed_data',
                            file_name='val.csv')
        train, test = self.split_dataset(df, test_size=0.3)
        self.save_dataframe(train, file_path='../processed_data',
                            file_name='train.csv')
        self.save_dataframe(test, file_path='../processed_data',
                            file_name='test.csv')

    @staticmethod
    def train_cb(df_path):
        df = pd.read_csv(df_path)
        X = df.drop('tgt', axis=1)
        y = df['tgt']
        train_data = Pool(data=X, label=y)
        params = config.CATBOOST_PARAMS
        model = CatBoostClassifier(**params)
        model.fit(train_data)
        return model
