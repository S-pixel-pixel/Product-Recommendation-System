# ========================= Streamlit Deployment for Product Rating Prediction =========================
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Product Recommendation - Rating Prediction App")

# Load best model and label encoders
with open("best_recommendation_model (1).pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.success("Model and encoders loaded successfully!")

uploaded_file = st.file_uploader("Upload a CSV file (must contain userId and productId columns)", type=["csv"])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_data.head())

        # Keep only necessary columns
        if not all(col in input_data.columns for col in ['userId', 'productId']):
            st.error("The uploaded file must contain 'userId' and 'productId' columns.")
        else:
            # Select only required columns
            input_data = input_data[['userId', 'productId']].copy()

            # Apply label encoders safely
            for col in ['userId', 'productId']:
                if col in encoders:
                    input_data[col] = input_data[col].map(
                        lambda x: encoders[col].transform([x])[0]
                        if x in encoders[col].classes_
                        else np.random.randint(0, len(encoders[col].classes_))
                    )

            # Predict ratings
            preds = model.predict(input_data[['userId', 'productId']])
            input_data['Predicted_Rating'] = preds

            st.success("Predictions Generated Successfully!")
            st.dataframe(input_data.head(20))
    except Exception as e:
        st.error(f"Error: {e}")

