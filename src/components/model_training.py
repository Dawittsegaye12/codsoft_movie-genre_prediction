import pickle
import joblib
from sklearn.model_selection import GridSearchCV
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

# Function to perform Grid Search and evaluate models
def grid_search_and_evaluate(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_accuracy = 0

    # Define parameter grids for each model
    param_grids = {
        'Naive Bayes': {'alpha': [0.01, 0.1, 1, 10]},
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200, 300]
        },
        'Support Vector Machine': {
            'C': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
    }

    # Loop through each model for GridSearchCV
    for name, model in models.items():
        print(f"Running GridSearchCV for {name}...")

        # Perform GridSearchCV
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)  # Train the model

        # Get the best model from the grid search
        best_grid_model = grid_search.best_estimator_

        # Make predictions using the best model
        y_pred = best_grid_model.predict(X_test)
        
        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Best Parameters: {grid_search.best_params_}")
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Save the best-performing model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_grid_model

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

    # Perform grid search, train, and evaluate models, getting the best model
    best_model = grid_search_and_evaluate(models, X_train, y_train, X_test, y_test)

    # Save the best model
    if best_model:
        joblib.dump(best_model, 'best_model.pkl')
        print("Best Model has been saved!")

if __name__ == '__main__':
    main()
