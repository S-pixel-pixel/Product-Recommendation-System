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

uploaded_file = st.file_uploader("Upload a CSV file (must contain userid and productid columns)", type=["csv"])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_data.head())

        # 🔧 Normalize column names
        input_data.columns = input_data.columns.str.lower()

        # Match encoders case-insensitively
        normalized_encoders = {k.lower(): v for k, v in encoders.items()}

        # Ensure required columns
        if not all(col in input_data.columns for col in ['userid', 'productid']):
            st.error("The uploaded file must contain 'userId' and 'productId' columns.")
        else:
                           
                display_data = input_data[['userid', 'productid']].head(20).copy()
                
                # Create encoded copy for prediction
                encoded_data = display_data.copy()
                
                # Apply encoders safely (Option A: Replace unseen IDs randomly)
                for col in ['userid', 'productid']:
                    if col in normalized_encoders:
                        le = normalized_encoders[col]
                        encoded_data[col] = encoded_data[col].map(
                            lambda x: le.transform([x])[0]
                            if x in le.classes_
                            else np.random.randint(0, len(le.classes_))
                        )
                
                # Ensure numeric
                encoded_data = encoded_data.fillna(0).astype(float)
                
                # Predict ratings
                preds = model.predict(encoded_data[['userid', 'productid']])
                
                # Add predictions to original display data
                display_data['Predicted_Rating'] = preds
                
                st.success("🎯 Predictions Generated Successfully!")
                st.dataframe(display_data.head(20))

    except Exception as e:
        st.error(f"Error: {e}")
