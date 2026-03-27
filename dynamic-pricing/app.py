from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []

        for feature in features:
            if feature == "season":
                val = encoder.transform([request.form[feature]])[0]
            else:
                val = float(request.form[feature])

            input_data.append(val)

        
        amazon_price = float(request.form['amazon_price'])
        prediction = model.predict([input_data])[0]
        best_price = amazon_price - prediction

        return render_template("index.html", result=f"🔥 Recommended Best Price: ₹ {round(best_price,2)}")

    except Exception as e:
        return render_template("index.html",
            result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)