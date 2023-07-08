import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# Load the dataset containing symptoms and diseases
m_df = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and target (y)
X = m_df.drop('prognosis', axis=1)
y = m_df['prognosis']

# Encode the target labels
le = preprocessing.LabelEncoder()
le.fit(y)
Y = le.transform(y)

# Create and train the decision tree classifier model
dtc = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=10, min_samples_leaf=24)
dtc.fit(X, Y)

# Dictionary mapping diseases to specialties
disease_specialty = [
    (["Vertigo", "Paroymsal Positional Vertigo"], "Otolaryngology"),
    (["AIDS", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Dengue", "Chicken pox", "Malaria", "Tuberculosis", "Typhoid"], "Infectious Disease"),
    (["Acne", "Drug Reaction", "Fungal infection", "Impetigo", "Psoriasis"], "Dermatology"),
    (["Alcoholic hepatitis", "Chronic cholestasis", "Jaundice", "Peptic ulcer disease", "Gastroenteritis"], "Gastroenterology"),
    (["Allergy"], "Allergy and Immunology"),
    (["Arthritis", "Osteoarthristis"], "Rheumatology"),
    (["Bronchial Asthma", "Pneumonia"], "Pulmonology"),
    (["Cervical spondylosis"], "Orthopedics"),
    (["Common Cold"], "Family Medicine"),
    (["Diabetes", "Hyperthyroidism", "Hypoglycemia", "Hypothyroidism"], "Endocrinology"),
    (["Dimorphic hemmorhoids(piles)", "Varicose veins"], "Colorectal Surgery"),
    (["GERD"], "Gastroenterology"),
    (["Heart attack"], "Cardiology"),
    (["Migraine", "Paralysis (brain hemorrhage)"], "Neurology"),
    (["Urinary tract infection"], "Urology"),
    (["Malaria"], "Infectious Disease"),
    (["Psoriasis"], "Dermatology"),
    (["Hypertension"], "Cardiology"),
    (["Impetigo"], "Dermatology")
]
# Function to predict disease and return specialty
def predict_disease(symptoms):
    input_vector = np.zeros(len(X.columns))
    for symptom in symptoms:
        if symptom in X.columns:
            input_vector[X.columns.get_loc(symptom)] = 1
    predicted_label = dtc.predict([input_vector])[0]
    predicted_disease = le.inverse_transform([predicted_label])[0]
    specialty = None
    for diseases, spec in disease_specialty:
        if predicted_disease in diseases:
            specialty = spec
            break
    return f"{predicted_disease}\nYou may have to visit a {specialty} doctor."
