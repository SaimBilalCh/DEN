import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Text preprocessing function
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

# Load and prepare dataset
def load_data():
    # For demonstration, we'll use a sample from the SMS Spam Collection dataset
    # In practice, you would load the actual dataset
    try:
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    except:
        # If unable to download, create a small sample dataset
        print("Unable to download dataset. Creating sample data...")
        data = {
            'label': ['ham', 'spam', 'ham', 'spam', 'ham'],
            'message': [
                'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C\'s apply 08452810075over18\'s',
                'Nah I don\'t think he goes to usf, he lives around here though',
                'Even my brother is not like to speak with me. They treat me like aids patent.',
                'I\'ve been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.',
                'As per your request \'Melle Melle (Oru Minnaminunginte Nurungu Vettam)\' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune'
            ]
        }
        df = pd.DataFrame(data)
    
    # Preprocess text
    df['processed_text'] = df['message'].apply(preprocess_text)
    
    # Map labels to binary (0 for ham, 1 for spam)
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

# Train models with TF-IDF features
def train_models_tfidf(df):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label_binary'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    
    # Evaluate models
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return tfidf_vectorizer, models, results, X_test_tfidf, y_test

# Main execution
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    
    print("Training models with TF-IDF features...")
    tfidf_vectorizer, models, results, X_test, y_test = train_models_tfidf(df)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save models and vectorizer for later use in Streamlit app
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(models['Logistic Regression'], 'spam_classifier.pkl')
    
    print("\nModels saved successfully!")