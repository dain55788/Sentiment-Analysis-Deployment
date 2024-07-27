from flask import Flask, render_template, request
import pickle
# Create pickle file instances
cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

# Create an Instance of the Flask Class
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('sentiment_analysis.html')

# Apply ML model
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        text = request.form.get('review_text')
    tokenized_text = cv.transform([text]) # X
    prediction = clf.predict(tokenized_text)

    # if the comment is 1, it should be a positive one
    prediction = 1 if prediction == 1 else -1
    return render_template("sentiment_analysis.html", prediction=prediction, text=text)

# Run the Flask Application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)