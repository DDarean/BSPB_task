import pandas as pd


def process_csv(df_path, payment_start_month_number=3, min_monthly_sum=100000):
    if not 1 < payment_start_month_number < 12:
        raise ValueError('Check month number')

    df = pd.read_csv(df_path)
    df = df[['id_client', 'day', 'tran_sum']]

    df['month'] = df['day'].apply(lambda x: pd.to_datetime(x).month)
    df['year'] = df['day'].apply(lambda x: pd.to_datetime(x).year)

    grouped = df.groupby(['id_client', 'year', 'month']).sum()
    grouped = grouped.reset_index()

    grouped['previous'] = grouped.groupby(['id_client', 'year'])[
        'tran_sum'].apply(
        lambda x: (x.shift(1, fill_value=0) >= min_monthly_sum) & (x >= min_monthly_sum))

    grouped['cashback'] = False

    for i in range(1, len(grouped)):
        if (grouped.loc[i, 'month'] >= payment_start_month_number) and \
                (grouped.loc[i, 'previous']) and not (grouped.loc[i - 1, 'cashback']):
            grouped.loc[i, 'cashback'] = True
        else:
            grouped.loc[i, 'cashback'] = False

    return grouped[['id_client', 'year', 'month', 'tran_sum', 'cashback']]


def cashback_summary(df, cashback_size=1000):
    summary = df.groupby(['year', 'month']).sum()[['cashback']]
    summary['cashback_rub'] = summary['cashback'] * cashback_size
    return summary
