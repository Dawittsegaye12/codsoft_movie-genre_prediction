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

