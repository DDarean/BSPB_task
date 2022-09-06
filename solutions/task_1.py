from copy import deepcopy
from typing import Tuple

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class Task1solver:
    """
    Task 1 solution
    Pipeline: process_df --> generate_dataset --> train_model --> predict

    Attributes:
        year: last available period for training (year)
        n_lags: number of previous days for feature generation
    """

    def __init__(self):
        self.year = 2020
        self.n_lags = 365 * 2
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=4.5)
        self.data = None
        self.featured_data = None

    def process_df(self, df_path: str) -> None:
        """
        Converts .csv file to pandas dataframe with required index and columns
        :param df_path: path to .csv file with training data
        :return: None
        """
        data = pd.read_csv(df_path)
        data.columns = ["date", "value"]
        data.set_index("date", inplace=True)
        data.index = pd.to_datetime(data.index)
        self.data = data

    @staticmethod
    def generate_lags(
        df: pd.DataFrame, n_lags: int, column_name="scaled"
    ) -> pd.DataFrame:
        """
        Creates n columns (features) with previous values for target column
        :param df: pandas dataframe
        :param n_lags: number of previous periods for features creation
        :param column_name: target column name
        :return: dataframe with new features
        """
        data = deepcopy(df)
        for n in range(1, n_lags + 1):
            data[f"day_lag{n}"] = data[column_name].shift(n)
        return data

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

    def generate_dataset(self) -> None:
        """
        Create dataframe and fit scaler for train/test data and empty rows for
        prediction range. Test data values set to zero to prevent data leakage
        :return: None
        """
        train_data = self.data[self.data.index.year <= self.year]
        self.scaler.fit(train_data["value"].values.reshape(-1, 1))
        train_data["scaled"] = self.scaler.transform(
            train_data["value"].values.reshape(-1, 1)
        )

        test_data = self.data.loc[self.data.index.year > self.year]
        test_data["scaled"] = 0

        daterange = pd.date_range(start="2022-01-01", end="2024-12-31")
        pred_data = pd.DataFrame(daterange)
        pred_data["value"] = 0
        pred_data["scaled"] = 0
        pred_data.columns = ["date", "value", "scaled"]
        pred_data.set_index("date", inplace=True)

        full_data = pd.concat([train_data, test_data, pred_data])

        self.featured_data = self.generate_lags(
            full_data, self.n_lags, "scaled"
        )
        self.featured_data.fillna(0, inplace=True)

    def train_model(self) -> None:
        """
        Fit model with train data
        :return: None
        """
        for_train = self.featured_data[
            self.featured_data.index.year <= self.year
        ][self.n_lags:].drop("value", axis=1)
        target_col = "scaled"
        x_train, y_train = self.target_split(for_train, target_col)
        self.model.fit(x_train.values, y_train.values.ravel())

    def predict(self) -> pd.DataFrame:
        """
        Make predictions using prediction for previous period as feature for
        next period
        :return: pandas dataframe with predictions
        """
        for_pred = self.featured_data[
            self.featured_data.index.year > self.year
        ].reset_index()

        for i in range(0, len(for_pred)):
            row = (
                for_pred.drop(["date", "value", "scaled"], axis=1)
                .iloc[i]
                .values.reshape(1, -1)
            )
            pred = self.model.predict(row)[0]
            for_pred.loc[i, "scaled"] = pred
            if i > self.n_lags - 1:
                lag_counter = self.n_lags - 1
            else:
                lag_counter = i
            for n in range(1, lag_counter + 2):  # n_lags + 1):
                for_pred.loc[i + 1, f"day_lag{n}"] = for_pred.loc[
                    i + 1 - n, "scaled"
                ]

        for_pred["prediction"] = self.scaler.inverse_transform(
            for_pred["scaled"].values.reshape(-1, 1)
        )

        predictions = for_pred[["date", "value", "prediction"]]
        predictions.set_index("date", inplace=True)
        return predictions
