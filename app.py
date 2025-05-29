import streamlit as st
import pickle
import pandas as pd
import numpy as np

import os
st.write("Current Working Directory:", os.getcwd())
st.write("Files in Working Directory:", os.listdir())
st.write("Files in DATA folder:", os.listdir("DATA"))

# Page configuration
st.set_page_config(page_title="Medical Recommendation System", layout="wide")

# Load the trained model and datasets
rf = pickle.load(open('rf_model.pkl', 'rb'))
description = pd.read_csv('DATA/description.csv')
precautions = pd.read_csv('DATA/precautions_df.csv')
medications = pd.read_csv('DATA/medications.csv')
diets = pd.read_csv('DATA/diets.csv')
workout = pd.read_csv('DATA/workout_df.csv')
symptoms_df = pd.read_csv('DATA/symtoms_df.csv')

symptoms_dict = {
    'itching': 0, 'skin rash': 1, 'nodal skin eruptions': 2, 'continuous sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint pain': 6, 'stomach pain': 7, 'acidity': 8, 'ulcers on tongue': 9, 'muscle wasting': 10, 'vomiting': 11,
    'burning micturition': 12, 'spotting urination': 13, 'fatigue': 14, 'weight gain': 15, 'anxiety': 16, 'cold hands and feets': 17,
    'mood swings': 18, 'weight loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches in throat': 22, 'irregular sugar level': 23,
    'cough': 24, 'high fever': 25, 'sunken eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30,
    'headache': 31, 'yellowish skin': 32, 'dark urine': 33, 'nausea': 34, 'loss of appetite': 35, 'pain behind the eyes': 36,
    'back pain': 37, 'constipation': 38, 'abdominal pain': 39, 'diarrhoea': 40, 'mild fever': 41, 'yellow urine': 42,
    'yellowing of eyes': 43, 'acute liver failure': 44, 'fluid overload': 45, 'swelling of stomach': 46, 'swelled lymph nodes': 47,
    'malaise': 48, 'blurred and distorted vision': 49, 'phlegm': 50, 'throat irritation': 51, 'redness of eyes': 52, 'sinus pressure': 53,
    'runny nose': 54, 'congestion': 55, 'chest pain': 56, 'weakness in limbs': 57, 'fast heart rate': 58, 'pain during bowel movements': 59,
    'pain in anal region': 60, 'bloody stool': 61, 'irritation in anus': 62, 'neck pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen legs': 68, 'swollen blood vessels': 69, 'puffy face and eyes': 70, 'enlarged thyroid': 71,
    'brittle nails': 72, 'swollen extremeties': 73, 'excessive hunger': 74, 'extra marital contacts': 75, 'drying and tingling lips': 76,
    'slurred speech': 77, 'knee pain': 78, 'hip joint pain': 79, 'muscle weakness': 80, 'stiff neck': 81, 'swelling joints': 82,
    'movement stiffness': 83, 'spinning movements': 84, 'loss of balance': 85, 'unsteadiness': 86, 'weakness of one body side': 87,
    'loss of smell': 88, 'bladder discomfort': 89, 'foul smell of urine': 90, 'continuous feel of urine': 91, 'passage of gases': 92,
    'internal itching': 93, 'toxic look (typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle pain': 97, 'altered sensorium': 98,
    'red spots over body': 99, 'belly pain': 100, 'abnormal menstruation': 101, 'dischromic patches': 102, 'watering from eyes': 103,
    'increased appetite': 104, 'polyuria': 105, 'family history': 106, 'mucoid sputum': 107, 'rusty sputum': 108, 'lack of concentration': 109,
    'visual disturbances': 110, 'receiving blood transfusion': 111, 'receiving unsterile injections': 112, 'coma': 113,
    'stomach bleeding': 114, 'distention of abdomen': 115, 'history of alcohol consumption': 116, 'fluid overload 1': 117, 'blood in sputum': 118,
    'prominent veins on calf': 119, 'palpitations': 120, 'painful walking': 121, 'pus filled pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin peeling': 125, 'silver like dusting': 126, 'small dents in nails': 127, 'inflammatory nails': 128,
    'blister': 129, 'red sore around nose': 130, 'yellow crust ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer disease',
    1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids (piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthritis', 5: 'Arthritis', 0: 'Paroxysmal Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# Helper function to retrieve disease details (description, precautions, etc.)
def helper(predicted_disease):
    try:
        desc = description[description['Disease'] == predicted_disease]['Description'].astype(str)
        desc = " ".join([w for w in desc])
    except IndexError:
        desc = "Description not available."

    try:
        pre = precautions[precautions['Disease'] == predicted_disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [col for col in pre.values]
    except IndexError:
        pre = ["Precautions not available."]
    
    try:
        med = medications[medications['Disease'] == predicted_disease]['Medication'].astype(str)
        med = [med for med in med.values]
    except IndexError:
        med = ["Medication not available."]
    
    try:
        die = diets[diets['Disease'] == predicted_disease]['Diet'].astype(str)
        die = [die for die in die.values]
    except IndexError:
        die = ["Diet not available."]
    
    try:
        wrkout = workout[workout['disease'] == predicted_disease]['workout'].astype(str)
        wrkout = [wrkout for wrkout in wrkout.values]
    except IndexError:
        wrkout = ["Workout not available."]
    
    return desc, pre, med, die, wrkout

# Format list of items into markdown bullet points
def format_list(items):
    flat_list = []
    for sublist in items:
        if isinstance(sublist, (list, np.ndarray)):
            for i in sublist:
                if i and str(i).strip() and str(i).lower() != 'nan':
                    flat_list.append(i)
        else:
            if sublist and str(sublist).strip() and str(sublist).lower() != 'nan':
                flat_list.append(sublist)
    return "\n".join([f"- {item}" for item in flat_list])

# Predict disease based on selected symptoms
def get_predicted_value(patient_Symtoms):
    input_vector = [0]* len(symptoms_dict)

    for item in patient_Symtoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            print(f"warning: symptom '{item}' not recognized!")

    input_vector = np.array(input_vector).reshape(1,-1)
    predicted_index = rf.predict(input_vector)[0]

    if isinstance(predicted_index, int):
        return diseases_list[predicted_index]
    else:
        return predicted_index

# Sidebar: app instructions
with st.sidebar:
    st.title("🩺 Medical AI Assistant")
    st.markdown("""
    **Purpose:**  
    Predicts diseases using AI based on your symptoms and provides:
    - 📖 Description  
    - ⚠️ Precautions  
    - 💊 Medications  
    - 🥗 Diet  
    - 🏋️ Workout  

    **Instructions:**  
    - Select symptoms you are experiencing.
    - Click **Recommend** to get your results.
    """)
    with st.expander("📝 View all available symptoms"):
        st.write(", ".join(sorted(symptoms_dict.keys())))

# Main title
st.title("Medical Recommendation System 🏥")
st.markdown("Get personalized health recommendations by selecting your symptoms below.")

# Use multiselect for symptom input
selected_symptoms = st.multiselect(
    "🔍 Select symptoms you are experiencing",
    options=sorted(symptoms_dict.keys())
)

# Button to recommend based on symptoms
if st.button("Recommend"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        predicted_disease = get_predicted_value(selected_symptoms)

        if not predicted_disease:
            st.error("⚠️ Could not predict a disease. Please check your symptom selection or try different symptoms.")
        else:
            st.markdown(f"## 🧾 Predicted Disease: `{predicted_disease}`")

            desc, pre, med, die, wrkout = helper(predicted_disease)

            st.markdown("### 📖 Disease Description")
            st.info(desc)

            st.markdown("### ⚠️ Precautions")
            st.success(format_list(pre))

            st.markdown("### 💊 Medication")
            st.warning(format_list(med))

            st.markdown("### 🥗 Diet Recommendations")
            st.success(format_list(die))

            st.markdown("### 🏋️ Workout Recommendations")
            st.info(format_list(wrkout))
