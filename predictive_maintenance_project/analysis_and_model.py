import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Инициализация session_state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        if st.button("Использовать встроенный датасет"):
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    if 'data' in locals() and data is not None:
        st.session_state.data = data

        # Предобработка данных
        data_clean = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
        le = LabelEncoder()
        data_clean['Type'] = le.fit_transform(data_clean['Type'])

        scaler = StandardScaler()
        numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        data_clean[numerical_features] = scaler.fit_transform(data_clean[numerical_features])

        # Разделение данных
        X = data_clean.drop(columns=['Machine failure'])
        y = data_clean['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение моделей
        if st.button("Обучить модели"):
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            }

            best_model_name = ""
            best_auc = 0
            plt.figure(figsize=(10, 8))

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)

                # ROC-кривая
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

                if auc > best_auc:
                    best_auc = auc
                    best_model_name = name
                    st.session_state.best_model = model

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC-кривые')
            plt.legend()
            st.pyplot(plt)

            st.success(f"Лучшая модель: {best_model_name} (AUC = {best_auc:.2f})")
            st.session_state.models = models

        # Оценка модели
        if st.session_state.best_model:
            model = st.session_state.best_model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.header("Результаты оценки")
            st.write(f"Accuracy: {accuracy:.2f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred)
            st.text(report)

            # Предсказания
            st.header("Предсказание по новым данным")
            with st.form("prediction_form"):
                st.write("Введите значения признаков:")

                col1, col2 = st.columns(2)
                with col1:
                    type_val = st.selectbox("Тип продукта", ["L", "M", "H"])
                    air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                    process_temp = st.number_input("Температура процесса [K]", value=310.0)

                with col2:
                    rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
                    torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                    tool_wear = st.number_input("Износ инструмента [min]", value=100)

                submit = st.form_submit_button("Предсказать")

                if submit:
                    # Преобразование введенных данных
                    input_data = pd.DataFrame({
                        'Type': [type_val],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotational_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                    })

                    # Преобразование категориальной переменной
                    input_data['Type'] = le.transform(input_data['Type'])

                    # Масштабирование
                    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                    # Предсказание
                    prediction = model.predict(input_data)
                    prediction_proba = model.predict_proba(input_data)[:, 1]

                    result = "Отказ" if prediction[0] == 1 else "Нет отказа"
                    st.success(f"Предсказание: {result}")
                    st.info(f"Вероятность отказа: {prediction_proba[0]:.2%}")


analysis_and_model_page()
