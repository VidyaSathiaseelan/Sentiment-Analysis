import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model("sentiment_model.keras")

# Function to predict sentiment
def predict_sentiment(text):
    max_length = 128  # Ensure this matches what was used in training
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    logits = model.predict(padded_sequences)
    predicted_class = np.argmax(logits, axis=1)[0]  # Get the class label
    
    sentiment_map = {
        0: ("Negative", "😞"),  
        1: ("Neutral", "😐"),  
        2: ("Positive", "😊")   
    }
    
    return sentiment_map.get(predicted_class, ("Unknown", "❓"))

# Streamlit UI Styling
st.markdown(
    """
    <style>
    html, body, {
        background-color: black !important;
        color: white !important;
    }

    /* Floating Smiley Animation */
    @keyframes floatSmiley {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0px); }
    }

    .smiley { 
        position: fixed;
        font-size: 50px;
        opacity: 0.2;
        animation: floatSmiley 3s infinite;
    }
    }

    /* Random Smiley Positions */
    .smiley1 { top: 5%; left: 5%; animation-duration: 4s; }
    .smiley2 { top: 15%; left: 40%; animation-duration: 6s; }
    .smiley3 { top: 25%; left: 70%; animation-duration: 5s; }
    .smiley4 { top: 40%; left: 20%; animation-duration: 7s; }
    .smiley5 { top: 55%; left: 80%; animation-duration: 6s; }
    .smiley6 { top: 70%; left: 10%; animation-duration: 5s; }
    .smiley7 { top: 85%; left: 50%; animation-duration: 4s; }
    .smiley8 { top: 90%; left: 90%; animation-duration: 8s; }
    </style>

    <div class="smiley smiley1">😊</div>
    <div class="smiley smiley2">😃</div>
    <div class="smiley smiley3">😁</div>
    <div class="smiley smiley4">😂</div>
    <div class="smiley smiley5">😍</div>
    <div class="smiley smiley6">🤩</div>
    <div class="smiley smiley7">😎</div>
    <div class="smiley smiley8">🥳</div>
    """,
    unsafe_allow_html=True
)

st.title("Sentiment Analysis with LSTM 🎭")
user_input = st.text_area("Enter a review:", "")

if st.button("Analyze"):
    if user_input.strip():
        sentiment_text, sentiment_emoji = predict_sentiment(user_input)
        st.write(f"### Sentiment: {sentiment_text} {sentiment_emoji}")
    else:
        st.warning("Please enter a review before analyzing!")
