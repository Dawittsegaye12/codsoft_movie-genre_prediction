import pickle
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load data from pickle files
def load_data(train_file, test_file):
    with open(train_file, 'rb') as f:
        train_df = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_df = pickle.load(f)
    return train_df, test_df

# Function to initialize models
def initialize_models():
    return {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Support Vector Machine': SVC(kernel='linear')
    }

# Function to train and evaluate models
def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_accuracy = 0

    # Loop through each model
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Make predictions

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Save the best-performing model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model

# Main execution
def main():
    # Load the preprocessed data
    train_df, test_df = load_data('training_data.pkl', 'testing_data.pkl')

    # Extract features and labels
    X_train = train_df['tfidf_features']
    y_train = train_df['genre_encoded']
    X_test = test_df['tfidf_features']
    y_test = test_df['genre_encoded']

    # Initialize models
    models = initialize_models()

    # Train and evaluate models, getting the best model
    best_model = train_and_evaluate_models(models, X_train, y_train, X_test, y_test)

    # Save the best model
    if best_model:
        joblib.dump(best_model, 'best_model.pkl')
        print("Best Model has been saved!")

if __name__ == '__main__':
    main()
