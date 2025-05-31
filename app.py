import streamlit as st
from presentation import presentation_page
from analysis_and_model import analysis_page  # создайте аналогичную функцию

pages = {
    "Анализ и модель": analysis_page,
    "Презентация": presentation_page,
}

st.sidebar.title("Навигация")
selected_page = st.sidebar.radio("Выберите страницу", list(pages.keys()))

# Вызов выбранной функции
pages[selected_page]()