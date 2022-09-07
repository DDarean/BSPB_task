import streamlit as st
from solutions.task_3 import Task3solver

month = st.number_input(label='Payment start month', min_value=2, max_value=12,
                        step=1, value=3)

min_payment = st.number_input(label='Min payment', value=100000, min_value=0,
                              step=1000)

cb_size = st.number_input(label='Cashback sum', value=1000, min_value=0,
                          step=100)

path_to_file = st.text_input(label='Path to data file',
                             value='data/task3.csv')

if st.button(label='calculate CB'):
    solver = Task3solver(month=3, min_payment=100000, cb_size=1000)
    solver.process_df(path_to_file)
    solver.make_clients_table()
    st.write('Ежемесячные траты клиентов (пример)')
    table = solver.clients_table[solver.clients_table['id_client'] == 0].drop(
        'year', axis=1)
    st.dataframe(table.head(5).style.format(formatter='{:.2f}'))
    solver.check_for_cashback()
    summary_table = solver.create_summary_table()
    st.dataframe(summary_table.style.format(formatter='{:.0f}'))

    csv = summary_table.to_csv().encode('utf-8')

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )
