from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

import solutions.config as config
from typing import Tuple

class Task2solver:
    """
    Task 2 solution
    """

    def __init__(self):
        self.RANDOM_STATE = config.RANDOM_STATE
        self.params = config.CATBOOST_PARAMS
        self.model = CatBoostClassifier(**self.params)

    @staticmethod
    def save_dataframe(
        df: pd.DataFrame, file_path=".", file_name="file.csv"
    ) -> None:
        """
        Save dataframe
        :param df: pandas dataframe
        :param file_path: path for saving
        :param file_name: new file name
        :return: None
        """
        path = Path(file_path).joinpath(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def target_split(
            df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Makes feature / target variable split for dataframe
        :param df: pandas dataframe
        :param target_col: name of target variable
        :return: pandas dataframes with features and labels
        """
        x = df.drop(columns=[target_col])
        y = df[[target_col]]
        return x, y

    def prepare_dataset(self, df_path: str) -> None:
        """
        Converts .csv file to pandas dataframe with required index and columns
        :param df_path: path to .csv file with training data
        :return: None
        """
        df = pd.read_csv(df_path)
        df = df.drop("Unnamed: 0", axis=1)
        train_test, val = train_test_split(
            df, test_size=0.1, random_state=self.RANDOM_STATE
        )
        self.save_dataframe(
            val, file_path="../processed_data", file_name="val.csv"
        )
        train, test = train_test_split(
            train_test, test_size=0.3, random_state=self.RANDOM_STATE
        )
        self.save_dataframe(
            train, file_path="../processed_data", file_name="train.csv"
        )
        self.save_dataframe(
            test, file_path="../processed_data", file_name="test.csv"
        )

    def train_cb(self, df_path: str) -> None:
        """
        Train CatBoost classifier
        :param df_path: path to processed .csv with train data
        :return: None
        """
        df = pd.read_csv(df_path)
        x = df.drop("tgt", axis=1)
        y = df["tgt"]
        train_data = Pool(data=x, label=y)
        self.model.fit(train_data)

    def random_search_train(self, df_path: str) -> None:
        """
        Train CatBoost classifier with best parameters found by random search
        :param df_path: path to processed .csv with train data
        :return: None
        """
        df = pd.read_csv(df_path)
        x, y = self.target_split(df, 'tgt')
        train_data = Pool(data=x, label=y)
        param_grid = config.CB_PARAM_GRID
        print(param_grid)
        self.model.grid_search(param_grid, train_data, refit=True, cv=5)