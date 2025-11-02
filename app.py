%pip install pandas
%pip install numpy
%pip install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
# Suppress the warnings that sometimes appear with model loading
warnings.filterwarnings('ignore')

# --- 1. Load the Model and Preprocessor ---
@st.cache_resource # Cache resource for faster loading
def load_assets():
    """Loads the trained ML model and preprocessor pipeline."""
    try:
        with open('preprocessor.pkl', 'rb') as file:
            preprocessor = pickle.load(file)
        with open('best_rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return preprocessor, model
    except FileNotFoundError:
        st.error("Error: Model or preprocessor files not found. Ensure 'preprocessor.pkl' and 'best_rf_model.pkl' are in the same directory.")
        return None, None

preprocessor, model = load_assets()

# --- 2. Streamlit App Structure ---
st.set_page_config(page_title="Startup Success Predictor", layout="wide")
st.title("💡 Startup Success Prediction Tool")
st.markdown("Use this tool to estimate the probability of a startup achieving a major exit (Acquisition/IPO).")

if model:
    # --- 3. Sidebar Inputs for Features ---
    st.sidebar.header("Input Startup Characteristics")
    
    # Define input fields based on the features used in training
    total_funding = st.sidebar.number_input(
        "Total Funding Raised (USD)",
        min_value=1000,
        max_value=1000000000, # 1 Billion
        value=10000000, # Default: $10 Million
        step=1000000,
        help="The total amount of equity funding the company has raised."
    )

    company_age = st.sidebar.slider(
        "Company Age (Years)",
        min_value=1,
        max_value=15,
        value=4,
        help="How long the company has been operating."
    )

    founders_count = st.sidebar.number_input(
        "Number of Founders",
        min_value=1,
        max_value=10,
        value=2
    )
    
    # Categorical features based on simulation
    industry = st.sidebar.selectbox(
        "Industry Sector",
        ('Software', 'Biotech', 'E-commerce', 'Hardware', 'Media', 'Other')
    )

    country = st.sidebar.selectbox(
        "Headquarters Country",
        ('USA', 'UK', 'India', 'Germany', 'Other')
    )

    prev_success = st.sidebar.selectbox(
        "Founder with Previous Successful Exit?",
        ('Yes', 'No')
    )
    
    # Convert binary input to numerical (1/0)
    previous_success_bin = 1 if prev_success == 'Yes' else 0

    # --- 4. Prediction Logic ---
    
    # Create the input DataFrame exactly as the model expects
    input_data = pd.DataFrame([{
        'Total_Funding_USD': total_funding,
        'Industry_Sector': industry,
        'Country': country,
        'Founders_Count': founders_count,
        'Previous_Success': previous_success_bin,
        'Company_Age_Years': company_age
    }])
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction on button click
    if st.button("Predict Success"):
        # Get the probability of the positive class (Success=1)
        prediction_proba = model.predict_proba(processed_data)[:, 1][0]
        prediction_percent = prediction_proba * 100
        
        # Display Results
        st.header("Prediction Result")
        
        st.metric(
            label="Predicted Success Probability", 
            value=f"{prediction_percent:.2f}%"
        )
        
        # Interpretation of the result
        if prediction_percent >= 60:
            st.success("High potential for success (Acquisition or IPO) based on current data.")
        elif prediction_percent >= 40:
            st.warning("Moderate potential. This startup warrants further due diligence.")
        else:
            st.error("Lower potential for major exit. Higher risk profile.")

        st.markdown("---")
        st.subheader("Input Review")
        st.write(input_data)
