import streamlit as st
from solutions.task_2 import Task2solver

st.markdown('### Задание 2')
st.write('Модель: CatBoost classifier  \n'
         'Обучение происходит в несколько этапов:  \n'
         '1. после подготовки датасета - начальное обучение и подбор гиперпараметров при помощи grid search и кросс-валидации  \n'
         '2. после этого проиходит отбор признаков и повторное обучение модели только на отобранных признаках  \n'
         '3. Для предсказаний модель обучается на всех доступных данных с параметрами, подобранными на предыдущем шаге  \n'
         '4. Предсказания доступны для сохранения в .csv формате')

path_to_train = st.text_input(label='Path to train data file',
                           value='../data/task2_train.csv')

path_to_submit = st.text_input(label='Path to data file for predictions',
                           value='data/task2_submit.csv')

if path_to_train:
    st.write('Начать обучение модели')
    if st.button(label='RUN'):
        solver = Task2solver()
        with st.spinner('Processing csv file'):
            solver.prepare_dataset(path_to_train)
        with st.spinner('Searching best parameters'):
            solver.grid_search_train()
        f1, precision, recall = solver.validate()
        st.write(f'Model\'s metrics  \n'
                 f'f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')
        solver.filter_top_features()
        with st.spinner('Retrain model on top features'):
            solver.update_model()
        f1_t, precision_t, recall_t = solver.validate()
        st.write(f'Metrics on top features and best parameters  \n'
                 f'f1: {f1_t:.4f}, precision: {precision_t:.4f}, recall: {recall_t:.4f}')
        with st.spinner('Making predictions'):
            preds = solver.retrain_and_predict(path_to_train,
                                          path_to_submit)
        st.write('Predictions')
        st.write(preds.head(5))

        st.write('Скачать предсказания')

        st.download_button(
            label="Download predictions as CSV",
            data=preds.to_csv(index=False).encode('utf-8'),
            file_name='submit1.csv',
            mime='text/csv'
        )
