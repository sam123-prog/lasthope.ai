from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["text"]
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        if prediction == 1:
            result = "Likely AI Generated 🤖"
        else:
            result = "Likely Human Written 👨"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)