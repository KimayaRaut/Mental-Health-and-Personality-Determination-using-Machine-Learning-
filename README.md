# Mental Health and Personality Prediction Web Application

## Overview

This project is a Flask-based web application designed to predict mental health and personality traits based on user-input text. The application uses an XGBoost model trained on the MBTI dataset to predict personality traits and a sentiment analysis model to assess mental health status from text.

- **Personality Prediction:** Utilizes an XGBoost model with 70% accuracy, trained on the MBTI dataset.
- **Mental Health Prediction:** Utilizes a sentiment analysis model with 95% accuracy to assess mental health status.

## Dataset

The dataset used for training the models is available on Kaggle: [MBTI Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)

## Technologies Used

- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** XGBoost (from scikit-learn)
- **Python Libraries:** pandas, numpy, scikit-learn, xgboost

## Installation

### Step 1: Clone the Repository

git clone https:[//github.com/yourusername/crop-forecasting.git
cd crop-forecasting](https://github.com/KimayaRaut/Crop-Forecasting-Using-Machine-Learning-Random-Forest-Classifier.git)

### Step 2: Create a Virtual Environment
Using virtualenv


```plaintext 
pip install virtualenv
```


```plaintext 
virtualenv venv
```

### Step 3: Activate the Virtual Environment
On Windows:


```plaintext 
venv\Scripts\
```


On MacOS/Linux:


```plaintext
source venv/bin/activate
```

### Step 4: Install the Required Packages
```plaintext 
pip install -r requirements.txt
```

### Step 5: Train the Models
Train the MBTI Model:
```plaintext 
python mbti_model.py
```
This will generate vectorizer.pkl, which is used for personality prediction.

Train the Sentiment Analysis Model:
```plaintext 
python sentiment_model.py
```
This will generate vectorizer1.pkl, which is used for mental health prediction.

### Step 6: Start the Flask Server
```plaintext 
python app.py
```
### Usage
Ensure the Flask server is running.

Open your web browser and navigate to http://127.0.0.1:5000.

Enter the text input to get predictions for personality traits and mental health status.

### File Structure
```plaintext
mental-health-personality-prediction/
├── app.py                      # Flask application
├── counter_vector.py           # Script for vectorization in the sentiment analysis model
├── mbti_model.py               # Script to train and save the XGBoost model for personality prediction
├── sentiment_model.py          # Script to train and save the XGBoost model for sentiment analysis
├── static/
│   └── css/
│       └── styles.css          # CSS file for styling
├── templates/
│   └── index.html              # HTML file for the frontend
├── vectorizer.pkl              # Trained XGBoost vectorizer for personality prediction
├── vectorizer1.pkl             # Trained XGBoost vectorizer for sentiment analysis
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```
