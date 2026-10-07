from flask import Flask, render_template, request
import pickle


# Create Flask application
app = Flask(__name__)


# Load trained ML model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""

    if request.method == "POST":

        # Get message from HTML form
        message = request.form["message"]

        # Convert message into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        result = model.predict(message_tfidf)[0]

        # Convert 0/1 into readable result
        if result == 1:
            prediction = "SPAM"
        else:
            prediction = "HAM"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)