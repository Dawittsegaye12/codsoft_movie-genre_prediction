import pandas as pd
import numpy as np
import re
import nltk
import pickle
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from textblob import TextBlob
from sklearn.metrics import classification_report
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data = pd.read_csv(r'/content/test_data_solution.txt.zip', delimiter=' ::: ', engine='python', header=None)
data.columns = ['ID', 'Title', 'Genre', 'Description']


data2= pd.read_csv(r'/content/train_data.txt.zip', delimiter=' ::: ', engine='python', header=None)
data2.columns = ['ID', 'Title', 'Genre', 'Description']

# Load spaCy's English tokenizer and lemmatizer
nlp = spacy.load("en_core_web_sm")

# Define file paths (update as necessary)
if data.shape[1] == 4 and data2.shape[1] == 4:
    df = pd.concat([data, data2], ignore_index=True)

# Drop rows with missing descriptions
df = df.dropna(subset=['Description'])

# 2. Feature Engineering

# Sentiment Label Creation
def get_sentiment(text):
    analysis = TextBlob(str(text))
    score = analysis.sentiment.polarity
    return 1 if score > 0 else (-1 if score < 0 else 0)

df['sentiment'] = df['Description'].apply(get_sentiment)

# Word Count
df['word_count'] = df['Description'].apply(lambda x: len(str(x).split()))

# Character Count
df['char_count'] = df['Description'].apply(lambda x: len(str(x)))

# 3. Label Encoding for Genre
encoder = LabelEncoder()
df['genre_encoded'] = encoder.fit_transform(df['Genre'])

# 4. Text Preprocessing

# Convert to Lowercase
df['Description'] = df['Description'].str.lower()

# Remove Special Characters and Punctuation
df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))

# Remove Stopwords
stop_words = set(nlp.Defaults.stop_words)

def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])

df['Description'] = df['Description'].apply(remove_stopwords)

# Tokenization and Lemmatization using spaCy
def lemmatize_text(text):
    doc = nlp(str(text))  # Process the text with spaCy
    return ' '.join([token.lemma_ for token in doc])

df['Description'] = df['Description'].apply(lemmatize_text)

# 5. Word Embedding (Bidirectional RNN)

# Prepare Text for Embedding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Description'])
sequences = tokenizer.texts_to_sequences(df['Description'])
X = pad_sequences(sequences, padding='post')

# Define the RNN model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=X.shape[1]),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model (using sentiment as the target for simplicity)
y = df['sentiment'].values
model.fit(X, y, epochs=5, batch_size=64, verbose=1)

# Get the word embedding output (this is the fixed-length vector for each description)
embeddings = model.predict(X)

# Add the embeddings as new features
df['embedding'] = embeddings.tolist()

# 6. Standardization (apply only to numerical columns like word_count and char_count)
scaler = StandardScaler()
df[['word_count', 'char_count']] = scaler.fit_transform(df[['word_count', 'char_count']])

# Final Dataset
logger.info("Final transformed dataset: ")
logger.info(df.head())

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save transformed data
with open('training_data.pkl', 'wb') as f:
    pickle.dump(train_df, f)

with open('testing_data.pkl', 'wb') as f:
    pickle.dump(test_df, f)

logger.info("Transformed data has been saved successfully.")

# Optionally, print classification report for model evaluation
logger.info("Classification Report:")
print(classification_report(y, model.predict(X).round()))
