import streamlit as st
import joblib
import re
import nltk

# Try to download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set page config
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

# Text preprocessing function (same as in training)
def preprocess_text(text):
    # Check if text is not a string (e.g., NaN)
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except:
        # If stopwords not available, continue without filtering
        pass
    
    # Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # If lemmatizer fails, use original tokens
        pass
    
    # Join tokens back to text
    return ' '.join(tokens)

# Load trained models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('spam_classifier.pkl')
        return vectorizer, model
    except:
        st.error("Model files not found. Please run the training script first.")
        return None, None

# Main app
def main():
    st.title("📱 SMS Spam Classifier")
    st.write("This app classifies SMS messages as either **spam** or **ham** (not spam) using Natural Language Processing.")
    
    # Load models
    vectorizer, model = load_models()
    
    if vectorizer and model:
        # Input text area
        user_input = st.text_area("Enter an SMS message to classify:", height=150, 
                                 placeholder="Type or paste an SMS message here...")
        
        if st.button("Classify"):
            if user_input:
                # Preprocess the input
                processed_text = preprocess_text(user_input)
                
                # Vectorize the text
                text_vectorized = vectorizer.transform([processed_text])
                
                # Make prediction
                prediction = model.predict(text_vectorized)
                prediction_proba = model.predict_proba(text_vectorized)
                
                # Display results
                if prediction[0] == 1:
                    st.error(f"**Result:** This message is **SPAM** 🚫")
                    st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
                else:
                    st.success(f"**Result:** This message is **HAM** (not spam) ✅")
                    st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
                
                # Show probability distribution
                st.subheader("Probability Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ham Probability", f"{prediction_proba[0][0]*100:.2f}%")
                with col2:
                    st.metric("Spam Probability", f"{prediction_proba[0][1]*100:.2f}%")
            else:
                st.warning("Please enter a message to classify.")
    
    # Add some information about the project
    with st.expander("About this project"):
        st.write("""
        This SMS Spam Classifier is built using:
        - **Natural Language Processing (NLP)** techniques for text preprocessing
        - **TF-IDF** for feature extraction
        - **Logistic Regression** for classification
        - **Streamlit** for the web interface
        
        The model was trained on the SMS Spam Collection dataset from UCI Machine Learning Repository.
        """)
        
    # Add footer
    st.markdown("---")
    st.caption("Built as part of the Digital Empowerment Network Machine Learning Task 05 - Text Classification with NLP")

if __name__ == "__main__":
    main()