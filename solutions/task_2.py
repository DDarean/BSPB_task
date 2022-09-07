from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import solutions.config as config


class Task2solver:
    """
    Task 2 solution
    """

    def __init__(self):
        self.RANDOM_STATE = config.RANDOM_STATE
        self.params = config.CATBOOST_PARAMS
        self.param_grid = config.CB_PARAM_GRID
        self.model = CatBoostClassifier(**self.params)
        self.train = None
        self.test = None
        self.top_features = None

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

    def prepare_dataset(self, df_path: str, test_size=0.3) -> None:
        """
        Converts .csv file to pandas dataframe with required index and columns
        :param df_path: path to .csv file with training data
        :param test_size: fraction of dataframe for test
        :return: None
        """
        df = pd.read_csv(df_path)
        df = df.drop("Unnamed: 0", axis=1)
        self.train, self.test = train_test_split(
            df, test_size=test_size, random_state=self.RANDOM_STATE
        )

    def train_fresh_cb(self, train_df: pd.DataFrame) -> None:
        """
        Retrain CatBoost classifier
        :return: None
        """
        self.model = CatBoostClassifier(**self.params)
        x, y = self.target_split(train_df, "tgt")
        train_data = Pool(data=x, label=y)
        self.model.fit(train_data)

    def grid_search_train(self) -> None:
        """
        Train CatBoost classifier with grid-search + cross validation
        :return: None
        """
        x, y = self.target_split(self.train, "tgt")
        train_data = Pool(data=x, label=y)
        param_grid = config.CB_PARAM_GRID
        self.model.grid_search(param_grid, train_data, refit=True, cv=5)
        self.params = self.model.get_params()

    def validate(self) -> Tuple[str, str, str]:
        """
        Calculate main metrics
        :return: f1-score, precision, recall
        """
        x, y = self.target_split(self.test, "tgt")
        pred = self.model.predict(x)
        f1 = f1_score(pred, y)
        precision = precision_score(pred, y)
        recall = recall_score(pred, y)
        return f1, precision, recall

    def filter_top_features(self, top_n=5) -> None:
        """
        Filter only top-n feature columns using feature importance
        :param top_n: quantity of features
        :return: None
        """
        if not self.top_features:
            self.top_features = self.model.get_feature_importance(
                prettified=True
            )["Feature Id"][0:top_n].values
        self.train = self.train[np.append(self.top_features, "tgt")]
        self.test = self.test[np.append(self.top_features, "tgt")]

    def update_model(self) -> None:
        """
        Retrain CatBoost classifier using grid search
        :return: None
        """
        self.model = CatBoostClassifier(**self.params)
        self.grid_search_train()

    def retrain_and_predict(
        self, train_path: str, pred_path: str
    ) -> pd.DataFrame:
        """
        Retrain classifier on best parameters and full train dataset and
        return predictions for test dataset
        :param train_path: path to train dataset
        :param pred_path: path to test dataset
        :return: dataframe with predictions for each row
        """
        train = pd.read_csv(train_path)
        train = train[np.append(self.top_features, "tgt")]
        self.train_fresh_cb(train)
        for_pred = pd.read_csv(pred_path)
        predictions = self.model.predict(for_pred[self.top_features])
        return pd.DataFrame(
            {"id": for_pred.iloc[:, 0].values, "prediction": predictions}
        )
