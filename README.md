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

```bash
git clone https://github.com/yourusername/mental-health-personality-prediction.git
cd mental-health-personality-prediction
Step 2: Create a Virtual Environment
Using virtualenv
bash
Copy code
pip install virtualenv
virtualenv venv
Using venv (Python 3.3+)
bash
Copy code
python -m venv venv
Step 3: Activate the Virtual Environment
On Windows
bash
Copy code
venv\Scripts\activate
On MacOS/Linux
bash
Copy code
source venv/bin/activate
Step 4: Install the Required Packages
bash
Copy code
pip install -r requirements.txt
Step 5: Train the Models
Train the MBTI Model
bash
Copy code
python mbti_model.py
This will generate mbti_model.pkl, which is used for personality prediction.

Train the Sentiment Analysis Model
bash
Copy code
python sentiment_model.py
This will generate sentiment_model.pkl, which is used for mental health prediction.

Step 6: Start the Flask Server
bash
Copy code
python app.py
Usage
Ensure the Flask server is running.
Open your web browser and navigate to http://127.0.0.1:5000.
Enter the text input to get predictions for personality traits and mental health status.
File Structure
plaintext
Copy code
mental-health-personality-prediction/
├── app.py                        # Flask application
├── mbti_model.py                 # Script to train and save the XGBoost model for personality prediction
├── sentiment_model.py            # Script to train and save the XGBoost model for sentiment analysis
├── templates/
│   └── index.html                # HTML file for the frontend
├── static/
│   └── css/
│       └── styles.css            # CSS file for styling
├── mbti_model.pkl                # Trained XGBoost model for personality prediction
├── sentiment_model.pkl           # Trained XGBoost model for sentiment analysis
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation```
