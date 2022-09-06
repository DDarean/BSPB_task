import streamlit as st

from solutions.task_1 import Task1solver

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot

path_to_df = st.text_input(label='Path to data file',
                             value='../data/task1_train.csv')

if path_to_df:
    if st.button(label='Run'):
        solver = Task1solver()
        solver.process_df(path_to_df)
        solver.generate_dataset()
        solver.train_model()
        predictions = solver.predict()

        datas = []

        value = go.Scatter(
            x=solver.data.index,
            y=solver.data.value,
            mode="lines",
            line={"dash": "dot"},
            name='Initial values',
            marker=dict(),
            opacity=0.8,
        )
        datas.append(value)

        prediction = go.Scatter(
            x=predictions.index,
            y=predictions.prediction,
            mode="lines",
            line={"dash": "dot"},
            name='Predictions',
            marker=dict(),
            opacity=0.8,
        )
        datas.append(prediction)

        layout = dict(
            title="Predictions",
            xaxis=dict(title="Time", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )

        fig = go.Figure(dict(data=datas, layout=layout))
        st.plotly_chart(fig)

