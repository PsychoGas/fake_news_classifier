import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Read the dataset from a file.
def read_data(file_name):
    try:
        df = pd.read_csv(file_name)
        labels = df.label
        return df, labels
    except FileNotFoundError:
        logging.error(f"File {file_name} not found.")
        raise

# Split the data into training and test sets.
def split_data(df, labels, test_size=0.2, random_state=7):
    return train_test_split(df['text'], labels, test_size=test_size, random_state=random_state)


# Convert a collection of raw documents to a matrix of TF-IDF features.
def vectorize_data(x_train, x_test):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    return tfidf_train, tfidf_test


# Train the PassiveAggressiveClassifier model.
def train_model(tfidf_train, y_train, max_iter=50):
    pac = PassiveAggressiveClassifier(max_iter=max_iter)
    pac.fit(tfidf_train, y_train)
    return pac


# Predict on the test set and calculate accuracy.
def evaluate_model(pac, tfidf_test, y_test):
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2)}%')
    return y_pred

# Print the confusion matrix.
def print_confusion_matrix(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

def print_predictions(y_test, y_pred, num_predictions=10):
    # Print the actual and predicted labels for the first num_predictions samples.
    print("-------------------------------------")
    for i in range(num_predictions):
        print(f"Actual: {y_test.iloc[i]}")
        print(f"Predicted: {y_pred[i]}")
        print("")

# ------------------------------------------------------------------------------------------
# PURELY OPTION -> Plot the actual and predicted labels for the first num_predictions samples.
# ------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
def plot_predictions(y_test, y_pred, num_predictions=10):
    """
    Plot the actual and predicted labels for the first num_predictions samples.
    """
    y_test_sample = y_test[:num_predictions]
    y_pred_sample = y_pred[:num_predictions]

    x = np.arange(len(y_test_sample))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, y_test_sample, width, label='Actual')
    rects2 = ax.bar(x + width/2, y_pred_sample, width, label='Predicted')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    fig.tight_layout()

    plt.show()


def main():
    df, labels = read_data('news.csv')
    x_train, x_test, y_train, y_test = split_data(df, labels)
    tfidf_train, tfidf_test = vectorize_data(x_train, x_test)
    pac = train_model(tfidf_train, y_train)
    y_pred = evaluate_model(pac, tfidf_test, y_test)
    print_confusion_matrix(y_test, y_pred)
    print_predictions(y_test, y_pred)
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()