

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('imdb_model.h5')


# Function to decode reviews
def decode_review(encoded_review):

    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):

    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#  Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')


user_input = st.text_area('Movie Review', placeholder='Type your review here...')


if st.button('Classify'):
    if user_input.strip():
        try:

            preprocessed_input = preprocess_text(user_input)


            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'


            st.write(f'**Sentiment:** {sentiment}')
            st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning('Please enter a valid movie review.')


st.write("review should be under 500 characters as max features = 500")

st.write("created by sanatan khemariya")
