import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from solutions.task_1 import Task1solver


def create_go_scatter(df: pd.DataFrame, column: str) -> go.Scatter:
    scatter = go.Scatter(
        x=df.index,
        y=df[column],
        mode="lines",
        line={"dash": "dot"},
        name=column,
        marker=dict(),
        opacity=0.8,
    )
    return scatter


def draw_chart(train: pd.DataFrame, pred: pd.DataFrame) -> None:
    x = []
    values = create_go_scatter(train, 'value')
    predictions = create_go_scatter(pred, 'prediction')
    x.append(values)
    x.append(predictions)

    layout = dict(
        title="Predictions",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = go.Figure(dict(data=x, layout=layout))
    st.plotly_chart(fig)


st.markdown('### Задание 1')
st.write('Модель обучается на данных с 2018 по 2020 год  \n'
         'Предсказания: с 2021 по 2024 год  \n'
         'По значениям 2021 вычисляются метрики качества  \n'
         'Остальные предсказания доступны для сохранения в .csv формате')

path_to_df = st.text_input(label='Path to data file',
                           value='data/task1_train.csv')

if path_to_df:
    st.write('Начать обучение модели')
    if st.button(label='Run'):
        solver = Task1solver()
        with st.spinner('Processing csv file'):
            solver.process_df(path_to_df)
        with st.spinner('Generating features'):
            solver.generate_dataset()
        with st.spinner('Training model'):
            solver.train_model()
        with st.spinner('Making predictions'):
            predictions = solver.predict()

        csv = predictions[predictions.index.year > 2021]
        csv = csv.reset_index()[['date', 'prediction']]

        st.write('Скачать предсказания')

        st.download_button(
            label="Download predictions as CSV",
            data=csv.to_csv(index=False).encode('utf-8'),
            file_name='submit1.csv',
            mime='text/csv'
        )

        draw_chart(solver.data, predictions)
