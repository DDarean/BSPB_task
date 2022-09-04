import streamlit as st

from solutions.task_3 import cashback_summary, process_csv

month = st.number_input(label='Payment start month', min_value=2, max_value=12,
                        step=1, value=3)

min_payment = st.number_input(label='Min payment', value=100000, min_value=0,
                              step=1000)

cb_size = st.number_input(label='Cashback sum', value=1000, min_value=0, step=100)

path_to_file = st.text_input(label='Path to data file',
                             value='../data/task3.csv')

if st.button(label='calculate CB'):
    processed_csv = process_csv(path_to_file, month, min_payment)
    summary = cashback_summary(processed_csv, cb_size)
    summary = summary.reset_index()
    st.dataframe(summary)

    csv = summary.to_csv().encode('utf-8')

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )
