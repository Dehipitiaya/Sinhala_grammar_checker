from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure the Gemini API
genai.configure(api_key="AIzaSyBMOttuWMdq7F6_5Ffb50SRYDKPiEaj6W4")

# Vectorizer and data storage
vectorizer = TfidfVectorizer()
grammar_rules = []  # Stores grammar rules for vectorization
rule_vectors = None  # Stores vectorized rules
vector_file = "vectorized_rules.pkl"  # File to store vectorized rules

# Load the grammar rules from the file and vectorize them
def load_grammar_rules():
    global grammar_rules, rule_vectors, vectorizer
    try:
        # Load grammar rules from file
        with open('grammar_rules.txt', 'r', encoding='utf-8') as file:
            grammar_rules = file.readlines()

        # Check if vectors already exist on disk
        if os.path.exists(vector_file):
            with open(vector_file, 'rb') as f:
                rule_vectors = pickle.load(f)
                vectorizer = TfidfVectorizer()  # Reinitialize vectorizer
                vectorizer.fit(grammar_rules)  # Fit vectorizer with grammar rules
        else:
            # Vectorize the grammar rules and save to disk
            vectorizer.fit(grammar_rules)
            rule_vectors = vectorizer.transform(grammar_rules)
            with open(vector_file, 'wb') as f:
                pickle.dump(rule_vectors, f)

    except Exception as e:
        print(f"Error loading grammar rules: {e}")

# Retrieve the most relevant grammar rules for the input text
def retrieve_relevant_rules(input_text, top_k=3):
    if rule_vectors is None:
        return []

    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, rule_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_rules = [grammar_rules[i] for i in top_indices]
    return relevant_rules

@app.route('/check-grammar', methods=['POST'])
def check_grammar():
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Retrieve relevant grammar rules
        relevant_rules = retrieve_relevant_rules(input_text)

        if not relevant_rules:
            return jsonify({"error": "Failed to retrieve relevant grammar rules"}), 500

        # Generate grammar correction using Gemini
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 512,
            "top_p": 0.95,
            "top_k": 40,
        }

        # Start chat and send the input text with relevant grammar rules as context
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )
        chat_session = model.start_chat()

        # Combine relevant rules into the prompt
        rules_context = "\n".join(relevant_rules)
        prompt = f"Here are the relevant grammar rules:\n{rules_context}\n\nPlease suggest corrections for the following Sinhala text in writing form and only respond with the Sinhala text: {input_text}"
        response = chat_session.send_message(prompt)

        # Process Gemini API response
        suggestions = response.text.strip() if response.text else "No suggestions provided."

        return jsonify({"suggestions": suggestions, "relevant_rules": relevant_rules}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_grammar_rules()
    app.run(debug=True)