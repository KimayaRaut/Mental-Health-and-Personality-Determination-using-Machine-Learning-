from flask import Flask, render_template, request
import numpy as np
import pickle
import counter_vector
from numpy.core.fromnumeric import var

model1 = pickle.load(open("pima.pickle.dat", "rb"))
model2 = pickle.load(open("pima.pickle1.dat", "rb"))
# model1 = pickle.load(open('model1.pkl', 'rb'))
# pickle.dump(vectorizer, open("vectorizer.pickle", "wb")) //Save vectorizer
# pickle.load(open("models/vectorizer.pickle", 'rb'))     // Load vectorizer
vectorizer = pickle.load(open("vectorizer.pickle", 'rb')) 
vectorizer1 = pickle.load(open("vectorizer1.pickle", 'rb')) 
encoder = pickle.load(open("encoder.pickle", 'rb')) 
app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    data1 = ""
    new_val =""
    if request.method=='POST':
        data1 = request.form['data1']
    new_data = [data1]
    arr = vectorizer.transform(new_data).toarray()
    arr1 = vectorizer1.transform(new_data).toarray()
    pred = model1.predict(arr)
    new_pred = encoder.inverse_transform(pred)
    pred1 = model2.predict(arr1)
    if pred1 == 0:
        new_val = "Stable"
    else:
        new_val = " Unstable"
    final_pred = "Presonality Is: " + str(new_pred) + "    And     Mental Health Statues: " + str(new_val)

    return render_template('index.html',var = final_pred )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
