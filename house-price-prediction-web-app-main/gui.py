import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocessing import preprocessing, word2vec, predict_keras

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():

    description=[ str(x) for x in request.form.values()]
    print("==========================================================================~~~~~~~~***********************")
    print(description)
    print("==========================================================================~~~~~~~~***********************")

    df=preprocessing(str(description))
    desc=word2vec(df["description"])
    predicted_price=predict_keras(desc)
    predicted_price = "{:.2f}".format(predicted_price)

    return render_template('index.html', prediction_text='The predicted house price is ${} CAD'.format(predicted_price))

if __name__ == "__main__":
    app.run(debug=True)