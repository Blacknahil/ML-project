# Spam Detection Web Application

This project is a simple web application for detecting spam messages using a machine learning model. The application is built using Flask and provides an interactive interface for users to input a message and get a prediction whether it is spam or ham along with the probability of being one.

## Project Structure Explanation

- `app.py`: Main Flask application file.
- `train_model.py`: Script to train the spam detection model and save it.
- `templates/`: Directory containing HTML templates.
  - `index.html`: Main HTML file for the user interface.
- `model.pkl`: Pickle file containing the trained machine learning model.
- `vectorizer.pkl`: Pickle file containing the vectorizer.

## How It Works?

1. **Data Loading and Preprocessing**:
   - The dataset `SMSSpamCollection` is loaded and preprocessed using `CountVectorizer` to convert text messages into numerical data.

2. **Model Training**:
   - A `MultinomialNB` (Naive Bayes) model is trained on the preprocessed data.
   - The trained model and vectorizer are saved to disk using `pickle`.

3. **Web Application**:
   - A Flask web application is created with an endpoint to render the HTML form and an endpoint to handle predictions.
   - Users can input a message, which is sent to the server for prediction.
   - The server processes the input through the saved model and returns the prediction and probability.

## Prerequisites

- Python 3.x
- Flask
- scikit-learn
- pandas

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Teklez/AI.git
   cd AI
2. Install the required python packages

    - Flask
        ```bash
        pip install flask 

    - scikit-learn
        ```bash
        pip install scikit-learn
    - pandas
        ```bash
        pip install pandas

## How to run our program?

# Go to the correct path which is spam_detection and run app.py

```bash
cd project/spam_detection
python3 app.py

