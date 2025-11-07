# ========================= Streamlit Deployment for Product Rating Prediction =========================
import streamlit as st
import pandas as pd
import pickle

st.title("Product Recommendation - Rating Prediction App")

# Load best model and label encoders
with open("best_recommendation_model (1).pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.write("Model and encoders loaded successfully!")

uploaded_file = st.file_uploader("Upload a CSV file (must contain userId and productId columns)", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("🧾 Uploaded Data Preview:")
    st.dataframe(input_data.head())

    # Drop unnecessary columns if present
    keep_cols = ['userId', 'productId']
    input_data = input_data[[col for col in keep_cols if col in input_data.columns]]

    # Apply encoders to categorical columns
    for col in input_data.select_dtypes(include=['object']).columns:
        if col in encoders:
            input_data[col] = input_data[col].map(
                lambda s: encoders[col].transform([s])[0] if s in encoders[col].classes_ else 0
            )

    # Ensure required columns exist
    if all(col in input_data.columns for col in ['userId', 'productId']):
        try:
            predictions = model.predict(input_data[['userId', 'productId']])
            input_data['Predicted_Rating'] = predictions
            st.success("Predictions Generated Successfully!")
            st.dataframe(input_data[['userId', 'productId', 'Predicted_Rating']].head(15))
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("The uploaded file must contain 'userId' and 'productId' columns.")


