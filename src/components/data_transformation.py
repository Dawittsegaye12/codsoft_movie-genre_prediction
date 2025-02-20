import pandas as pd
import re
import spacy
import numpy as np
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from gensim.models import Word2Vec
import logging

# Initialize Spacy NLP
nlp = spacy.load("en_core_web_sm")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Transformers
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.clean_text(text) for text in X]

    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            sentiment = self.get_sentiment(text)
            word_count = len(text.split())
            char_count = len(text)
            features.append([sentiment, word_count, char_count])
        return np.array(features)

    @staticmethod
    def get_sentiment(text):
        score = TextBlob(str(text)).sentiment.polarity
        return 1 if score > 0 else (-1 if score < 0 else 0)

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self.text_to_vector(text) for text in X])

    def text_to_vector(self, text):
        words = text.split()
        vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(128)

# Model Training
def build_model(input_shape, num_classes):
    model = Sequential([ 
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load Data and Train Word2Vec
def load_data_and_train_w2v(train_path, test_path):
    df1 = pd.read_csv(test_path, delimiter=' ::: ', engine='python', header=None, names=['ID', 'Title', 'Genre', 'Description'])
    df2 = pd.read_csv(train_path, delimiter=' ::: ', engine='python', header=None, names=['ID', 'Title', 'Genre', 'Description'])
    df = pd.concat([df1, df2], ignore_index=True)
    df.dropna(subset=['Description'], inplace=True)
    corpus = [text.split() for text in df['Description'].tolist()] + [text.split() for text in df['Title'].tolist()]
    w2v_model = Word2Vec(sentences=corpus, vector_size=128, window=5, min_count=1, sg=1, workers=4)
    return df, w2v_model

# Full Pipeline
def create_pipeline(w2v_model):
    pipeline = Pipeline(steps=[
        ('preprocessing', TextPreprocessor()),  # Text preprocessing
        ('embedding', EmbeddingTransformer(w2v_model)),  # Word embedding (on raw text)
        ('scaler', StandardScaler()),  # Feature scaling
        ('model', build_model(input_shape=128, num_classes=10))  # Update num_classes accordingly
    ])
    return pipeline

# Run the entire pipeline
def run_pipeline(train_path, test_path):
    logger.info("Loading data and training Word2Vec...")
    df, w2v_model = load_data_and_train_w2v(train_path, test_path)

    # Extract text data (Description) and target labels (Genre)
    X = df['Description'].values  # Use raw text (descriptions)
    y = df['Genre'].values  # Assuming 'Genre' is your target variable

    logger.info("Creating pipeline...")
    pipeline = create_pipeline(w2v_model)

    logger.info("Training model...")
    pipeline.fit(X, y)

    # Save the model, encoder, and scaler if necessary
    # e.g., save the trained model to disk using pickle, joblib, or other formats

    return pipeline


if __name__ == "__main__":
    train_path = "C:\Users\SOOQ ELASER\movie_genre_prediction\dataset\train_data.txt.zip"
    test_path = "C:\Users\SOOQ ELASER\movie_genre_prediction\dataset\test_data_solution.csv"
    pipeline = run_pipeline(train_path, test_path)
