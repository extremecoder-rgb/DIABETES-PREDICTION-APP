import streamlit as st
import pandas as pd
import pickle
import sklearn 
import os

st.title("Diabetes Prediction App")

st.write("This app predicts the likelihood of diabetes based on user input data. Please fill in the form below to get started.")

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models')
model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')

try:
    with open(model_path, "rb") as f:
        model, preprocessor = pickle.load(f)
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
    st.error("Model file not found. Please ensure the model is trained and saved correctly.")
    st.stop()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

questions = {
    "Age": (
        "Age Category",
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80 or older",
    ),
    "Sex" : ("Sex", "Female", "Male"),
    "HighChol": ("High Cholesterol", "High", "Low"), 
    "CholCheck": ("Cholesterol Check in 5 Years", "Yes", "No"),
    "Mass": ("Mass (pounds)", 100, 300, 150),
    "Height": ("Height (inches)", 60, 80, 65),
    "Smoker": ("Smoked at least 5 packs (100 cigarets) in your life", "Yes", "No"),  
    "Stroke": ("Ever Had a Stroke", "Yes", "No"), 
    "HeartDiseaseorAttack": ("Ever Had Heart Disease or Heart Attack", "Yes", "No"),  
    "PhysActivity": ("Physical Activity", "Yes", "No"),  
    "Fruits": ("Consume Fruits 1 or more times per day", "Yes", "No"), 
    "Veggies": ("Consume Vegetables 1 or more times per day", "Yes", "No"),  
    "HvyAlcoholConsump": ("Heavy Alcohol Consumption", "Yes", "No"),  
    "AnyHealthcare": ("Any Healthcare Coverage", "Yes", "No"), 
    "GenHlth": (
        "How Would You Rate Your General Health",
        "Excellent",  
        "Very Good",
        "Good",
        "Fair",
        "Poor",
    ),
    "MentHlth": ("Number of days with bad mental health in past 30 days", 0, 30, 0),
    "PhysHlth": ("Number of days with bad physical health in past 30 days", 0, 30, 0),
    "DiffWalk": ("Difficulty Walking", "Yes", "No"),
    "HighBP": ("High Blood Pressure", "Yes", "No"),
    "NoDocbcCost": ("Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?", "Yes", "No")

}

def show_question():
    key = list(questions.keys())[st.session_state.current_question]
    
    st.subheader(questions[key][0])
    
    if key in ["Mass", "Height", "MentHlth", "PhysHlth"]:
        user_input = st.slider("", questions[key][1], questions[key][2], questions[key][3], key=key)
    elif key in ["GenHlth", "Education", "Income", "Age"]:
        user_input = st.selectbox("", options=questions[key][1:], index=0, key=key)
    elif key in ["HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "DiffWalk"]:
        user_input = st.radio("", options=[questions[key][1], questions[key][2]], index=0, key=key)
    else:
        user_input = st.radio("", options=[questions[key][1], questions[key][2]], index=0, key=key)
    
    st.session_state.user_inputs[key] = user_input
    
    if st.button("Next", key=questions[key][0]+ "button"):
        st.session_state.current_question += 1
        if st.session_state.current_question < len(questions):
            show_question()
        else:
            st.session_state.current_question = 0
            st.success("Thank you for your input!")
            st.session_state.submit_button_pressed = True
            st.balloons()
            
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "submit_button_pressed" not in st.session_state:
    st.session_state.submit_button_pressed = False
if "risk_calculated" not in st.session_state:
    st.session_state.risk_calculated = False

if st.session_state.current_question < len(questions):
    show_question()
    
if st.session_state.submit_button_pressed:
    
    input_data = pd.DataFrame([st.session_state.user_inputs])
    st.write(input_data)
 
    input_data["Sex"] = input_data["Sex"].map({"Female": 0, "Male": 1})
    input_data["HighChol"] = input_data["HighChol"].map({"High": 1, "Low": 0})
    input_data["CholCheck"] = input_data["CholCheck"].map({"Yes": 1, "No": 0})
    input_data["Smoker"] = input_data["Smoker"].map({"Yes": 1, "No": 0})
    input_data["Stroke"] = input_data["Stroke"].map({"Yes": 1, "No": 0})
    input_data["HeartDiseaseorAttack"] = input_data["HeartDiseaseorAttack"].map({"Yes": 1, "No": 0})
    input_data["PhysActivity"] = input_data["PhysActivity"].map({"Yes": 1, "No": 0})
    input_data["Fruits"] = input_data["Fruits"].map({"Yes": 1, "No": 0})
    input_data["Veggies"] = input_data["Veggies"].map({"Yes": 1, "No": 0})
    input_data["HvyAlcoholConsump"] = input_data["HvyAlcoholConsump"].map({"Yes": 1, "No": 0})
    input_data["AnyHealthcare"] = input_data["AnyHealthcare"].map({"Yes": 1, "No": 0})
    input_data["DiffWalk"] = input_data["DiffWalk"].map({"Yes": 1, "No": 0})
  
    input_data["GenHlth"] = input_data["GenHlth"].map(
        {
            "Excellent": 1,
            "Very Good": 2,
            "Good": 3,
            "Fair": 4,
            "Poor": 5,
        }
    )
    
    input_data["Age"] = input_data["Age"].map(
        {
            "18-24": 1,
            "25-29": 2,
            "30-34": 3,
            "35-39": 4,
            "40-44": 5,
            "45-49": 6,
            "50-54": 7,
            "55-59": 8,
            "60-64": 9,
            "65-69": 10,
            "70-74": 11,
            "75-79": 12,
            "80 or older": 13,
        }
    )
    # NUMERIC
    input_data["BMI"] = (input_data["Mass"]*0.45359237) / ((input_data["Height"] * 0.0254) ** 2)
    input_data.drop(columns=["Mass", "Height"])
    input_data["BMI"] = input_data["BMI"].round(2)
    
    input_data["MentHlth"] = input_data["MentHlth"].astype(int)
    input_data["PhysHlth"] = input_data["PhysHlth"].astype(int)
    
    try:
        processed_data = preprocessor.transform(input_data)
    except Exception as e:
        st.error(
            f"Error: An error occurred during preprocessing.  This may be due to the input data not matching the format the model expects.  Please check your answers. Error Details: {e}"
        )
        st.stop()
    
    try:
        prediction = model.predict_proba(processed_data)[:, 1]
        risk_per = prediction[0] * 100
        st.session_state.risk_calculated = True
    except Exception as e:
        st.error(
            f"Error: An error occurred during prediction.  This may be due to the input data not matching the format the model expects.  Please check your answers. Error Details: {e}"
        )
        st.stop()
    st.session_state.user_inputs = {}
    st.session_state.current_question = 0

if st.session_state.risk_calculated:
    st.subheader("Diabetes Risk Assessment Result")
    st.write(f"Your estimated risk of diabetes is: {risk_per:.2f}%")

   
    if risk_per < 15:
        st.write("Based on your answers, your risk is low.")
    elif risk_per < 30:
        st.write("Based on your answers, your risk is moderate.")
    else:
        st.write(
            "Based on your answers, your risk is elevated. It is recommended to consult with a healthcare professional."
        )