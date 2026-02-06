from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load trained model & vectorizer once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        # Prevent empty input
        if len(text) == 0:
            result = "Please enter some text first 🙂"
        else:
            text_vec = vectorizer.transform([text])
            prediction = model.predict(text_vec)[0]

            if prediction == 1:
                result = "Likely AI Generated 🤖"
            else:
                result = "Likely Human Written 👨"

    return render_template("index.html", result=result)


if __name__ == "__main__":

    # For Render / deployment
    port = int(os.environ.get("PORT", 10000))

    # debug=False for production
    app.run(host="0.0.0.0", port=port, debug=True)
