import joblib
import pandas as pd
import streamlit as st


# 4. Function to load the model to make a prediction
@st.cache_resource
def load_prediction_model():
  # Replace 'model.pkl' with the actual path to your trained model file
  model = joblib.load('heart_disease_risk_model.pki')
  return model


# Page Configuration
st.set_page_config(
    page_title='Cardiovascular Health Predictor', page_icon='❤️', layout='centered'
)

st.title('❤️ Coronary Health Risk Prediction App')
st.markdown(
    'Enter the patient details below to evaluate the cardiovascular risk'
    ' prediction.'
)

st.sidebar.header('Patient Parameters')

# 3. Streamlit input widgets for the specified columns
age_years = st.number_input(
    'Age (Years)',
    min_value=29.56,
    max_value=64.92,
    value=50.0,
    step=0.1,
    help='Patient age in years',
)

gender = st.sidebar.selectbox(
    'Gender',
    options=[0, 1],
    format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)',
)

smoke = st.sidebar.selectbox(
    'Smoking Status',
    options=[0, 1],
    format_func=lambda x: 'Non-Smoker (0)' if x == 0 else 'Smoker (1)',
)

alco = st.sidebar.selectbox(
    'Alcohol Intake',
    options=[0, 1],
    format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)',
)

active = st.sidebar.selectbox(
    'Physical Activity',
    options=[0, 1],
    format_func=lambda x: 'Inactive (0)' if x == 0 else 'Active (1)',
)

# --- CONVERT CATEGORICAL TO NUMBERS AND STORE WITH INPUT DATA ---
gender_val = 1 if gender == 'Male' else 0
smoke_val = 1 if smoke == 'Smoker' else 0
alco_val = 1 if alco == 'Yes' else 0
active_val = 1 if active == 'Active' else 0

# Create the final encoded input DataFrame matching your model's expected features
input_df = pd.DataFrame({
    'age_years': [age_years],
    'gender': [gender_val],
    'smoke': [smoke_val],
    'alco': [alco_val],
    'active': [active_val]
})

# Display the stored numerical input data in the app UI
st.subheader('Processed Numerical Input Data')
st.dataframe(input_df)


if st.button('Run Prediction', type='primary'):
  try:
    model = load_prediction_model()
    prediction = model.predict(input_df)
    prediction_proba = (
        model.predict_proba(input_df)
        if hasattr(model, 'predict_proba')
        else None
    )

    st.markdown('---')
    st.subheader('Prediction Result')

    if prediction[0] == 1:
      st.error(
          '**High Risk:** The model predicts the presence of cardiovascular'
          ' disease (`cardio = 1`).'
      )
    else:
      st.success(
          '**Low Risk:** The model predicts no cardiovascular disease (`cardio ='
          ' 0`).'
      )

    if prediction_proba is not None:
      risk_prob = prediction_proba[0][1] * 100
      st.info(f'Estimated Cardiovascular Disease Probability: **{risk_prob:.2f}%**')

  except FileNotFoundError:
    st.error(
        "Model file 'model.pkl' not found. Please ensure your trained model is"
        ' placed in the working directory.'
    )
  except Exception as e:
    st.error(f'An error occurred during prediction: {e}')
