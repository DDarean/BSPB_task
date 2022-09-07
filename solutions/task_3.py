import pandas as pd


class Task3solver:
    """
    Task 3 solution.
    Pipeline:
        process .csv file -> make clients table with monthly total ->
        check if cashback conditions are met -> summary table

    Attributes:
        month: first month for cashback payment
        min_payment: minimum spending required for cashback
        cb_size: monthly cashback payment
    """
    def __init__(self, month: int, min_payment: int, cb_size: int):
        if not 1 < month < 12:
            raise ValueError('Check month number')
        self.month = month
        self.min_payment = min_payment
        self.cb_size = cb_size
        self.data = None
        self.clients_table = None
        self.cashback_table = None

    def process_df(self, df_path: str) -> None:
        """
        Converts .csv file to pandas dataframe with required index and columns
        :param df_path: path to .csv file with training data
        :return: None
        """
        self.data = pd.read_csv(df_path)
        self.data = self.data[['id_client', 'day', 'tran_sum']]

    def make_clients_table(self) -> None:
        """
        Create summary table with monthly total for each client
        :return: None
        """
        clients_table = self.data[:]
        clients_table['month'] = clients_table['day'].apply(
            lambda x: pd.to_datetime(x).month)
        clients_table['year'] = clients_table['day'].apply(
            lambda x: pd.to_datetime(x).year)

        clients_table = clients_table.groupby(
            ['id_client', 'year', 'month']).sum()
        clients_table['tran_sum'] = clients_table['tran_sum'].round(2)
        self.clients_table = clients_table.reset_index()

    def check_for_cashback(self) -> None:
        """
        Check if cashback conditions are met and create monthly summary table
        with cashback flag for each client
        :return: None
        """
        cb_table = self.clients_table[:]
        cb_table['previous'] = cb_table.groupby(['id_client', 'year'])[
            'tran_sum'].apply(
            lambda x: (x.shift(1, fill_value=0) >= self.min_payment) & (
                        x >= self.min_payment))
        cb_table['cashback'] = False

        for i in range(1, len(cb_table)):
            if (cb_table.loc[i, 'month'] >= self.month) and \
                    (cb_table.loc[i, 'previous']) and not (
                    cb_table.loc[i - 1, 'cashback']):
                cb_table.loc[i, 'cashback'] = True
            else:
                cb_table.loc[i, 'cashback'] = False
        self.cashback_table = cb_table

    def create_summary_table(self):
        """
        Calculate cashback cases and monthly total based on cashback size
        :return: summary table with calculated total
        """
        data = self.cashback_table[:]
        summary = data.groupby(['year', 'month']).sum()[['cashback']]
        summary['cashback_rub'] = summary['cashback'] * self.cb_size
        summary.columns = ['Количество клиентов', 'Итого выплаты, тыс. руб.']
        return summary
