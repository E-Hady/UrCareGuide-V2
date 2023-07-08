from flask import Flask, request, jsonify
from machine_learning_model import predict_disease

# Flask app
app = Flask(__name__)

# Load the symptom data
data = pd.read_csv('unique_symptoms.csv')

base_list = []
check_list = []
asked_list = []
features = {}
# Function to select rows based on selected symptoms

def select_rows(data, base_list):
    mask = data.apply(lambda x: all(symptom in x.values.astype(str) for symptom in base_list), axis=1)
    selected_rows = data[mask]
    return selected_rows

# Welcome message
welcome_message = "Welcome to the Chatbot. Please enter 'yes' or 'no' for the following questions. Enter 'Exit' to end the chat."

# Webhook endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    request_data = request.get_json()
    response = ""

    # Extract the necessary information from the request JSON
    intent = request_data['queryResult']['intent']['displayName']
    parameters = request_data['queryResult']['parameters']

    if intent == 'start':
        response = welcome_message

    elif intent == 'exit':
        symptoms = parameters.get('symptoms', [])
        if symptoms:
            response = predict_disease(symptoms)
        else:
            response = "Please provide the symptoms for prediction."

    elif intent == 'yes':
        symptom = parameters.get('symptom')
        if symptom:
            # Process the 'yes' response from the user
            base_list.append(symptom)
            asked_list.append(symptom)
            features[symptom] = 1
            check_list = [symptom for symptom in check_list if symptom not in base_list]
            # Generate the appropriate response for the next question or prompt
        else:
            response = "Please provide a symptom for the 'yes' response."

    elif intent == 'no':
        symptom = parameters.get('symptom')
        if symptom:
            # Process the 'no' response from the user
            asked_list.append(symptom)
            features[symptom] = 0
            check_list = [symptom for symptom in check_list if symptom not in base_list]
            # Generate the appropriate response for the next question or prompt
        else:
            response = "Please provide a symptom for the 'no' response."

    else:
        response = "Sorry, I didn't understand that."

    # Create the webhook response
    webhook_response = {
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [response]
                }
            }
        ]
    }

    return jsonify(webhook_response)

if __name__ == '__main__':
    app.run(debug=True)
