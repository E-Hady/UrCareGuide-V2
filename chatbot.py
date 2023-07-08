import pandas as pd
from machine_learning_model import predict_disease

# Load the CSV data containing symptoms and disease information
data = pd.read_csv('unique_symptoms.csv')

# Function to select rows from the data based on selected symptoms
def select_rows(data, base_list):
    mask = data.apply(lambda x: all(symptom in x.values.astype(str) for symptom in base_list), axis=1)
    selected_rows = data[mask]
    return selected_rows

# Function to ask questions and collect user responses
def chatbot():
    base_list = []
    asked_list = []

    # Welcome message
    print("Welcome to the Chatbot. Please enter 'yes' or 'no' for the following questions.")
    print("Enter 'Exit' to end the chat.")

    while True:
        selected_rows = select_rows(data, base_list)

        symptom_counts = selected_rows.iloc[:, 1:].stack().value_counts()
        check_list = [symptom for symptom in symptom_counts.index if symptom not in asked_list and symptom not in base_list]

        for symptom in check_list:
            answer = input(f"Do you have {symptom}? (yes/no) ")
            asked_list.append(symptom)
            if answer.lower() == 'yes':
                base_list.append(symptom)
        
        # Perform prediction using the machine learning model based on the selected symptoms
        prediction = predict_disease(base_list)
        
        # Generate response based on the predicted disease
        
        # End the chat if 'Exit' is entered
        if answer.lower() == 'exit':
            print("Exiting the chatbot.")
            break

# Call the chatbot function to start the chat
chatbot()
