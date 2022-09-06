import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score
from solutions.task_2 import Task2solver

if st.button(label='RUN'):
    df = pd.read_csv('processed_data/test.csv')
    x_test = df.drop('tgt', axis=1)
    y_test = df['tgt']
    a = Task2solver()
    a.train_cb('processed_data/train.csv')
    st.write(f'f1-score: {f1_score(y_test, a.model.predict(x_test)):.4f}')
    