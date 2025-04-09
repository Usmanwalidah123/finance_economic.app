import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.title("\ud83d\udcca Stock Index Classifier")
st.markdown("Enter financial indicators to predict the stock index (e.g. S&P500, Nasdaq, etc.)")

@st.cache_data
def load_and_train():
    df = pd.read_csv("finance_economics_dataset.csv")

    # Encode Stock Index
    le = LabelEncoder()
    df["Stock Index"] = le.fit_transform(df["Stock Index"])

    # Drop Date
    df = df.drop(columns=["Date"])

    # Split
    X = df.drop(columns=["Stock Index"])
    y = df["Stock Index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return model, X.columns, le, cm

model, feature_names, le, cm = load_and_train()

# Dynamic inputs
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict Stock Index"):
    prediction = model.predict(input_df)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    st.success(f"\ud83c\udf1f Predicted Stock Index: **{predicted_label}**")

# Show confusion matrix
st.subheader("\ud83d\udcc9 Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax)
st.pyplot(fig)
