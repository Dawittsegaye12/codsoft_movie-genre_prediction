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

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Convert `Description` into numerical vectors using TF-IDF to capture word importance.

3. **Word Embeddings**
   - Utilize `word2vec`, `GloVe`, or `BERT embeddings` to capture contextual meaning from `Description`.

4. **Sentiment Analysis Features**
   - Encode `Sentiment_Label` as a numerical feature (0 for Negative, 1 for Positive).

5. **Lexical and Syntactic Features**
   - `word_count`: Represents sentence complexity.
   - `char_count`: Captures description length.

6. **N-grams Features**
   - Generate bigrams and trigrams from `Description` for better context representation.

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

---
### Author
**Dawit Tsegaye**

