# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import re
import string
import nltk
import pandas as pd
# import os
import spacy
import contractions
import requests
import json
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from translate import Translator
from scipy.sparse import csr_matrix, hstack
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spacy.cli import download


# # Load the pickled files
# save_path = '/Users/huiyee/Downloads/Study/Year3Sem2/FYP_project/backend_copy'

# os.makedirs(save_path, exist_ok=True)

# with open('tfidf_matrix.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# # Load the model from the pickle file
# with open('hybrid_model.pkl','rb') as file:
#     loaded_model_dict = pickle.load(file)

# # Extract the scaler and classifier
# loaded_scaler = loaded_model_dict['scaler']
# best_lr_classifier = loaded_model_dict['classifier']
# tokenizer = loaded_model_dict['tokenizer']

# # Load the Keras model
# best_model_rnn = load_model('sentiment_classifier_rnn.h5')

# Define preprocessing functions
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = expand_contractions(text)  # Expand contractions
    text = transform_chat_words(text)  # Transform chat words
    text = translate_to_english(text)  # Translate to English
    text = remove_urls(text)  # Remove URLs
    text = remove_html_tags(text)  # Remove HTML tags
    text = remove_punctuation(text)  # Remove punctuation
    text = remove_digits(text)  # Remove digits
    text = remove_stopwords(text)  # Remove stopwords
    tokens = word_tokenize(text)  # Tokenize
    tokens = remove_special_symbols(tokens)  # Remove special characters
    tokens = remove_empty_tokens(tokens)  # Remove empty tokens
    tokens = lemmatize_text(tokens)  # Lemmatize tokens
    text = ' '.join(tokens)  # Join tokens back into text
    return text

# Function to expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Chat words mapping dictionary
chat_word_mapping = {
    # Your chat word mappings here
    "afaik": "as far as i know",
    "afk": "away from keyboard",
    "asap": "as soon as possible",
    "atk": "at the keyboard",
    "atm": "at the moment",
    "a3": "anytime, anywhere, anyplace",
    "bak": "back at keyboard",
    "bbl": "be back later",
    "bbs": "be back soon",
    "bfn": "bye for now",
    "b4n": "bye for now",
    "brb": "be right back",
    "brt": "be right there",
    "btw": "by the way",
    "b4": "before",
    "b4n": "bye for now",
    "cu": "see you",
    "cul8r": "see you later",
    "cya": "see you",
    "faq": "frequently asked questions",
    "fc": "fingers crossed",
    "fwiw": "for what it's worth",
    "fyi": "for your information",
    "gal": "get a life",
    "gg": "good game",
    "gn": "good night",
    "gmta": "great minds think alike",
    "gr8": "great!",
    "g9": "genius",
    "ic": "i see",
    "icq": "i seek you (also a chat program)",
    "ilu": "ilu: i love you",
    "imho": "in my honest/humble opinion",
    "imo": "in my opinion",
    "iow": "in other words",
    "irl": "in real life",
    "kiss": "keep it simple, stupid",
    "ldr": "long distance relationship",
    "lmao": "laugh my ass off",
    "lol": "laughing out loud",
    "ltns": "long time no see",
    "l8r": "later",
    "mte": "my thoughts exactly",
    "m8": "mate",
    "nrn": "no reply necessary",
    "oic": "oh i see",
    "pita": "pain in the ass",
    "prt": "party",
    "prw": "parents are watching",
    "rofl": "rolling on the floor laughing",
    "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my ass off",
    "sk8": "skate",
    "stats": "your sex and age",
    "asl": "age, sex, location",
    "thx": "thank you",
    "ttfn": "ta-ta for now!",
    "ttyl": "talk to you later",
    "u2": "you too",
    "u4e": "yours for ever",
    "wb": "welcome back",
    "wtf": "what the fuck",
    "wtg": "way to go!",
    "wuf": "where are you from?",
    "w8": "wait...",
    "7k": "sick:-d laughter",
}

# Function to transform chat words
def transform_chat_words(text):
    for chat_word, expanded_form in chat_word_mapping.items():
        text = text.replace(chat_word, expanded_form)
    return text

def translate_to_english(text, lang='en'):
    try:
        if lang != 'en':
            translator = Translator(to_lang="en")
            translation = translator.translate(text)
        else:
            translation = text
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return None

# Function to remove URLs
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

# Function to remove HTML tags
def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]*>', '', text)
    return clean_text

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to remove digits
def remove_digits(text):
    return re.sub(r'\d+', '', text)

# Function to remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to remove special symbols
def remove_special_symbols(tokens):
    cleaned_tokens = []
    for token in tokens:
        pattern = r'[^a-zA-Z0-9\s]'
        cleaned_token = re.sub(pattern, '', token)
        cleaned_tokens.append(cleaned_token)
    return cleaned_tokens

# Function to remove empty tokens
def remove_empty_tokens(tokens):
    return [token for token in tokens if token.strip()]

# Function to lemmatize text using spaCy
# download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
def lemmatize_text(tokens):
    doc = nlp(' '.join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens

def make_predictions(user_input):
    # Preprocess the input
    preprocessed_input = preprocess_text(user_input)
    
    # Convert input to TF-IDF features
    input_tfidf = vectorizer.transform([preprocessed_input])
    
    # Pad the input for RNN model
    input_seq = tokenizer.texts_to_sequences([preprocessed_input])
    input_pad = pad_sequences(input_seq, maxlen=100, padding='post') 
    
    # Predict sentiment using the RNN model
    sentiment_predictions_prob = best_model_rnn.predict(input_pad)
    sentiment_predictions = np.argmax(sentiment_predictions_prob, axis=1)
    
    # Convert sentiment predictions to sparse matrix
    sentiment_features = csr_matrix(sentiment_predictions).reshape(-1, 1)
    
    # Stack TF-IDF features with sentiment predictions
    input_with_sentiment = hstack([input_tfidf, sentiment_features])
    
    # Scale the features
    input_with_sentiment_scaled = loaded_scaler.transform(input_with_sentiment)
    
    # Predict depression status using the Logistic Regression model
    depression_prediction = best_lr_classifier.predict(input_with_sentiment_scaled)
    return depression_prediction

def get_recommendation(user_input):
    api_key = "sk-OVIALiHKN7GNrlV700CfB1C838494081966fEd3eD64e3eCf"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
 
    prompt = f"""
        f"I have predicted signs of depression based on user input: '{user_input}'. 
        Provide some helpful recommendations to overcome."
    """
    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
 
    response = requests.post("https://api.wlai.vip/v1/chat/completions", headers=headers, json=data)
 
    try:
        response = requests.post("https://api.wlai.vip/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()
 
        # Check if 'choices' key is in the response
        if 'choices' in response_data:
            recommendation = response_data["choices"][0]["message"]["content"]
        else:
            st.error(f"Unexpected API response structure: {json.dumps(response_data, indent=2)}")
            recommendation = "No recommendations available due to unexpected API response."
 
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        st.error(f"Response content: {response.text if response else 'No response'}")
        recommendation = "Error in getting recommendations."
   
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {str(e)}")
        st.error(f"Response content: {response.text if response else 'No response'}")
        recommendation = "Error in getting recommendations."
 
    return recommendation

def main():
    st.markdown('<div class="main-title" >Depression Prediction and Recommendation App</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">This app predicts the likelihood of depression based on text input. If depression is predicted, it provides recommendations using ChatGPT model.</div>', unsafe_allow_html=True)
    
    # CSS styles
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.5rem;
                color: #4B0082;
                font-weight: 700;
                text-align: center;
            }
            .description {
                font-size: 1.25rem;
                color: #333;
                text-align: center;
                margin-bottom: 2rem;
            }
            .result {
                font-size: 1.25rem;
                color: #333;
                font-weight: 600;
            }
            # .recommendation {
            #     font-size: 1rem;
            #     color: #0073e6;
            # }
            .input-area, .predict-button {
                text-align: center;
                margin: 1rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True)

    # Layout for input and button
    user_input = st.text_area("Enter your text here:", 
                              placeholder="Share your thoughts and feelings here...", 
                              height=200, max_chars=500, key="input_text")
    if st.button("Predict"):
        if user_input:
            try:
                depression_prediction = make_predictions(user_input)
                predicted_class = "depression" if depression_prediction[0] == "depression" else "not depression"
                emoji = "🫂" if predicted_class == "depression" else "😊"
                st.markdown(
                    f'<div class="result">The model predicts that the text indicates signs of <b style="color: red;">{predicted_class}</b>.</div>', 
                    unsafe_allow_html=True
                )                
                st.markdown(f'<div class="emoji">{emoji}</div>', unsafe_allow_html=True)

                if predicted_class == "depression":
                    st.markdown('<div class="recommendation">Here are some recommendations for you:</div>', unsafe_allow_html=True)
                    recommendation = get_recommendation(user_input)
                    st.markdown(f'<div class="recommendation">{recommendation}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.write("Please enter some text to get a prediction.")

if __name__ == "__main__":
    main()
