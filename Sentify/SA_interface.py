import streamlit as st
import pickle
import time
from preprocessing import preprocess_text

# Load pre-trained SVM model and TF-IDF vectorizer
@st.cache_data()
def load_model():
    with open('svm_tfidf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return model, tfidf_vectorizer


# Function to preprocess text
def preprocess_input(text):
    preprocessed_text = preprocess_text(text)
    return preprocessed_text

# Function to predict sentiment using TF-IDF features
def predict_sentiment(text, model, tfidf_vectorizer):

    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Vectorize text using the TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([preprocessed_text])
    
    # Make predictions using the model
    sentiment_code = model.predict(text_vectorized)[0]
    
    # Map sentiment codes to labels and emojis
    if sentiment_code == 1:
        return "Positive"
    elif sentiment_code == 0:
        return "Neutral"
    elif sentiment_code == -1:
        return "Negative"

# Streamlit UI
st.title('Sentify')

# Load pre-trained model and TF-IDF vectorizer
model, tfidf_vectorizer = load_model()

# Input text box
text_input = st.text_area('Enter a sentence:', '')

# Button to predict sentiment
if st.button('Predict'):
    if text_input:
        # Predict sentiment
        sentiment = predict_sentiment(text_input, model, tfidf_vectorizer)
        
        # Display sentiment
        st.write('Sentiment:', sentiment)
        
        # Add effect for positive sentiment
        if 'Positive' in sentiment:
            # Display smiling emoji with increased size
            st.markdown('<p style="font-size:48px;">😊</p>', unsafe_allow_html=True)
            time.sleep(1)  # Pause to display the emoji for a moment
        elif 'Neutral' in sentiment:
            st.markdown('<p style="font-size:48px;">😐</p>', unsafe_allow_html=True)
            time.sleep(1)
        elif 'Negative' in sentiment:
            st.markdown('<p style="font-size:48px;">😞</p>', unsafe_allow_html=True)
            time.sleep(1)
    else:
        st.write('Please enter a sentence.')
