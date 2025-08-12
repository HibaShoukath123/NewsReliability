import re
import joblib
import pickle as pk
import gensim
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Define route for homepage (root URL)
@app.route('/', methods=["GET", "POST"])
def reliablity():
    input_pred = None  # Initialize variable for prediction result
    result = None      # Initialize variable for the result message

    if request.method == "POST":
        text1 = request.form.get("text")  # Get input text from the form
        input_text = text1
        input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)  # Clean the text (remove non-alphabetic characters)

        # Load the pre-trained Word2Vec model and the SVM model
        filename = r'svm_model2.pkl'  # Corrected the file extension
        model = gensim.models.Word2Vec.load('word2vec_model.bin')  # Load the Word2Vec model
        #loaded_model = pk.load(open(filename))  # Load the SVM model
        loaded_model = joblib.load(filename)

        # Tokenize the input text and create the feature vector for prediction
        input_tokens = input_text.lower().split()  # Tokenize the input text
        input_vector = np.zeros((1, model.vector_size))  # Initialize the vector with zeros

        # Summing the Word2Vec embeddings for the input tokens
        for token in input_tokens:
            if token in model.wv:  # Check if the token is in the Word2Vec model
                input_vector += model.wv[token]  # Add the token embedding to the vector

        # Normalize the vector by dividing by the number of tokens
        input_vector /= len(input_tokens) if len(input_tokens) > 0 else 1

        # Make prediction with the SVM model
        input_pred = loaded_model.predict(input_vector)
        input_pred = input_pred.astype(int)  # Ensure the prediction is in integer form

        # Based on the prediction, set the result message
        if input_pred[0] == 1:
            result = 'Review is Positive'
        else:
            result = "Review is Negative"

    # Render the result on the HTML page
    return render_template('index.html', prediction_text='News Reliability Analysis: {}'.format(result))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
