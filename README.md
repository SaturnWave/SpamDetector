# Spam Classifier using Naive Bayes

This project is a simple spam classifier that uses the Naive Bayes algorithm to classify text messages as either spam or not spam (ham). The dataset used for this classifier is the "Youtube01-Psy" dataset.

## Overview

This code provides a basic implementation of a spam classifier using Python and the scikit-learn library. It goes through the following steps:

1. Loading the dataset: The code reads the dataset from a CSV file (Youtube01-Psy.csv).

2. Preprocessing the data: It performs basic data exploration, including checking for missing values, calculating class distribution, and displaying some sample comments.

3. Text Vectorization: The text data is transformed into numerical features using the CountVectorizer and TF-IDF (Term Frequency-Inverse Document Frequency) techniques.

4. Model Training: The Multinomial Naive Bayes classifier is used to train the model on the transformed data.

5. Model Evaluation: The code evaluates the model's performance using 5-fold cross-validation and calculates accuracy and confusion matrix on the test set.

6. Predictions: The trained model is used to predict whether new comments are spam or not.

## Dependencies

To run this code, you need the following dependencies:

- Python 3
- scikit-learn
- pandas
- numpy
- nltk

You can install the required Python packages using pip:

```bash
pip install scikit-learn pandas numpy nltk

Additionally, you need to download the NLTK English stopwords by running the following Python code:import nltk
nltk.download('stopwords')


Usage
Clone the repository to your local machine.

Install the required dependencies (if not already installed).

Run the Python script spam_classifier.py. It will load the dataset, preprocess the data, train the model, and evaluate its performance.

After running the script, you'll see the accuracy and confusion matrix for the spam classifier.

Results
The spam classifier provides accuracy and confusion matrix results, indicating how well the model performs in classifying comments as spam or not spam.

Testing
The script also includes a section for testing the classifier on new comments. You can add new comments to the new_comments list and run the script to see how the model classifies them.

Feel free to use and extend this code for your own spam classification tasks.
