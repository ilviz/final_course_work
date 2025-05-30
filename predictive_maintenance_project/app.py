import streamlit as st

pages = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py",
}

st.sidebar.title("Навигация")
selected_page = st.sidebar.radio("Выберите страницу", list(pages.keys()))

with open(pages[selected_page], "r", encoding="utf-8") as file:
    exec(file.read())