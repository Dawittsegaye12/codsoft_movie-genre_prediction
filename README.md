# Movie Genre Classification

## Project Overview
This project aims to build a **Machine Learning model** that predicts the genre(s) of a movie based on its **plot summary**. The dataset consists of movie descriptions and their corresponding genres, making it a **multi-label text classification problem**.

## Dataset
- **Source:** [Kaggle - Genre Classification Dataset IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- **Features:**
  - `title`: Movie title
  - `summary`: Movie plot summary (text-based feature)
  - `genres`: Labels (multi-label classification)
- **Types of Datasets:**
  - IMDb Dataset
  - Movie Dataset
  - Merged Dataset

## Approach
We will use **Natural Language Processing (NLP) techniques** and **Machine Learning classifiers** to predict movie genres from text summaries. The key steps include:

### 1. Data Preprocessing
- Cleaning text (removing stopwords, punctuation, special characters)
- Tokenization
- Feature extraction using:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - **Word Embeddings** (Word2Vec, GloVe, or FastText)

### 2. Model Selection
We will experiment with multiple classification algorithms:
- **Naïve Bayes**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Transformer-based models** (BERT, DistilBERT) [optional]

### 3. Model Training & Evaluation
- Train multiple models using **TF-IDF and word embeddings**
- Evaluate performance using metrics like **accuracy, precision, recall, and F1-score**
- Use **multi-label classification metrics** (e.g., Jaccard similarity, Hamming loss)

## Experiment Tracking and Version Control
### **DVC (Data Version Control)**
- **Why?** Keeps track of data versions, pipelines, and ML models.
- **Planned Usage:**
  - Track dataset versions
  - Manage preprocessed data and features
  - Version control for trained models

### **MLflow (Experiment Tracking & Model Registry)**
- **Why?** Logs parameters, metrics, and trained models.
- **Planned Usage:**
  - Track hyperparameter tuning
  - Log model performance and comparisons
  - Store trained models for deployment

## Project Structure
```
movie_genre_classification/
│-- data/                 # Raw and processed datasets
│-- notebooks/            # Jupyter notebooks for EDA & modeling
│-- src/
│   ├── preprocessing.py  # Data cleaning & feature extraction
│   ├── model_training.py # Training and evaluation scripts
│   ├── infer.py          # Inference script
│-- models/               # Saved models
│-- dvc.yaml              # DVC pipeline configuration
│-- mlruns/               # MLflow experiment tracking
│-- requirements.txt      # Project dependencies
│-- README.md             # Project documentation
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd movie_genre_classification
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up **DVC** and **MLflow**:
   ```sh
   dvc init
   mlflow ui  # Starts MLflow tracking UI
   ```
4. Run the training script:
   ```sh
   python src/model_training.py
   ```

## Future Enhancements
- Implement **deep learning-based models** (LSTMs, Transformers)
- Deploy model using **Flask/FastAPI** as a REST API
- Integrate **streamlit-based UI** for genre prediction

---
Feel free to contribute and suggest improvements! 🚀

# Movie Genre Classification

## Project Overview
This project aims to classify movies into their respective genres based on their plot summaries. The dataset contains movie descriptions along with metadata such as title, sentiment labels, word count, and character count. Various machine learning techniques are employed to process and analyze the text data for accurate genre classification.

## Dataset Description
The dataset consists of the following columns:
- **ID**: Unique identifier for each movie.
- **Title**: Movie title.
- **Genre**: The actual genre of the movie.
- **Description**: A textual summary of the movie plot.
- **Sentiment_Label**: Sentiment classification of the description (Positive/Negative).
- **word_count**: The number of words in the description.
- **char_count**: The number of characters in the description.

## Feature Engineering
To improve model performance, the following feature engineering techniques are applied:
1. **Text Preprocessing**
   - Convert text to lowercase.
   - Remove special characters, punctuation, and stopwords.
   - Tokenization and lemmatization/stemming.

3. **Word Embeddings**
   - Utilize `bidirectional RNN` to capture contextual meaning from `Description`.

4. **Sentiment Analysis Features**
   - Encode `Sentiment_Label` as a numerical feature (0 neutral, -1 for Negative, 1 for Positive).

5. **Lexical and Syntactic Features**
   - `word_count`: Represents sentence complexity.
   - `char_count`: Captures description length.


7. **Genre One-Hot Encoding**
   - Convert `Genre` into a numerical format using one-hot encoding.

## Feature Selection
To improve efficiency, we apply the following feature selection techniques:
1. **Remove Redundant Features**
   - Drop `ID` since it does not contribute to genre classification.

2. **Statistical Tests for Feature Importance**
   - Apply **Chi-Square Test** or **Mutual Information** to identify relevant words in `Description` for classification.

3. **Dimensionality Reduction**
   - Use **PCA (Principal Component Analysis)** or **t-SNE** to reduce feature dimensions when using TF-IDF or word embeddings.

## Implementation
The project is implemented using Python with libraries such as:
- `pandas` for data processing
- `scikit-learn` for feature extraction and modeling
- `nltk` and `spaCy` for text preprocessing
- `tensorflow` or `transformers` for deep learning-based approaches

## Conclusion
This project explores various natural language processing (NLP) techniques to classify movie genres based on textual descriptions. By leveraging different feature engineering and selection strategies, the model aims to improve accuracy and generalization.

# Functionality of the Code

## 1. Library Imports and Initialization

- **Libraries**:
  - **Pandas**: Handles dataframes (CSV data).
  - **Spacy**: Used for advanced NLP tasks.
  - **TextBlob**: For sentiment analysis.
  - **Scikit-learn**: For preprocessing, building pipelines, and training models.
  - **TensorFlow**: For creating a neural network.
  - **Gensim**: Used to train Word2Vec for word embeddings.

- **Logging**: Configures logging to monitor the flow of operations.

---

## 2. Custom Transformers

### `TextPreprocessor`
- **Purpose**: A custom transformer that cleans text by converting it to lowercase and removing non-alphanumeric characters.
- **Functions**:
  - `fit`: Required by the scikit-learn pipeline (not used here).
  - `transform`: Applies the `clean_text` method on each entry in the dataset.
  - `clean_text`: Converts text to lowercase and removes special characters.

### `FeatureExtractor`
- **Purpose**: Extracts additional features like sentiment, word count, and character count from the text.
- **Functions**:
  - `fit`: Required by the scikit-learn pipeline (not used here).
  - `transform`: Calculates sentiment (positive, negative, or neutral), word count, and character count for each text entry.
  - `get_sentiment`: Uses `TextBlob` to calculate sentiment polarity.

### `EmbeddingTransformer`
- **Purpose**: Converts raw text into word embeddings using a trained Word2Vec model.
- **Functions**:
  - `fit`: Required by the scikit-learn pipeline (not used here).
  - `transform`: Uses Word2Vec to convert each text into word embeddings and returns the average of word vectors as the text's representation.
  - `text_to_vector`: Converts text into word vectors and averages them.

---

## 3. Model Training (`build_model`)

- **Purpose**: Builds a simple neural network model for multi-class classification (e.g., genre prediction).
- **Architecture**:
  - Input layer (128 units).
  - Hidden layer (64 units).
  - Output layer (number of classes = genres).
  - Softmax activation for multi-class classification.
  - Uses Adam optimizer and sparse categorical cross-entropy loss function.

---

## 4. Load Data and Train Word2Vec (`load_data_and_train_w2v`)

- **Purpose**: Loads training and test data, trains a Word2Vec model on the corpus of `Description` and `Title`.
- **Functions**:
  - Loads data from CSV files using Pandas.
  - Tokenizes `Description` and `Title` into words.
  - Trains a Word2Vec model using Gensim on the tokenized data.

---

## 5. Create the Full Pipeline (`create_pipeline`)

- **Purpose**: Creates a complete pipeline for preprocessing, feature extraction, and model training.
- **Pipeline Steps**:
  1. **Text Preprocessing**: Cleans the text.
  2. **Word Embedding**: Converts text to word embeddings using the trained Word2Vec model.
  3. **Scaling**: Standardizes the feature vectors.
  4. **Model**: A neural network is used to predict the target labels.

---

## 6. Run the Pipeline (`run_pipeline`)

- **Purpose**: Executes the entire workflow by loading data, training Word2Vec, building the pipeline, and training the model.
- **Steps**:
  - Loads data and trains Word2Vec.
  - Extracts text data (`Description`) and target labels (`Genre`).
  - Creates the pipeline and fits it to the data.
  - Optionally saves the model, encoder, and scaler.

---

## 7. Main Execution

- **Purpose**: The entry point of the script, which triggers the pipeline.
- **Steps**:
  - Defines the paths to the training and test data.
  - Calls `run_pipeline()` to start the entire process.

---

## Summary of Functionality

1. **Data Loading**: Loads training and test data from CSV files.
2. **Text Preprocessing**: Cleans and tokenizes the text data (descriptions and titles).
3. **Word2Vec Training**: Trains a Word2Vec model on the `Description` and `Title` columns.
4. **Feature Extraction**: Extracts additional features like sentiment, word count, and character count.
5. **Word Embedding**: Converts text into word embeddings using the trained Word2Vec model.
6. **Model Training**: Trains a neural network on the transformed data to predict the target labels (genres).
7. **Pipeline Execution**: The entire process is orchestrated using scikit-learn's pipeline mechanism, which combines the steps in a sequence and trains the model.

---

Let me know if you need more details or adjustments!

---
### Author
**Dawit Tsegaye**

