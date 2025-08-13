import os, re
import joblib
import numpy as np
from flask import Flask, request, render_template


from gensim.models import Word2Vec



import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.data.path.append(os.path.expanduser('~/nltk_data'))

STOPWORDS = set(stopwords.words('english'))
LEMM = WordNetLemmatizer()
NON_ALPHA = re.compile(r'[^a-zA-Z]')

def preprocess_tokens(text: str):
    """Regex clean → lowercase → tokenize → remove stopwords/short tokens → lemmatize."""
    text = NON_ALPHA.sub(' ', (text or '')).lower().strip()
    toks = text.split()
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    toks = [LEMM.lemmatize(t) for t in toks]
    return toks

def to_doc_vec(tokens, w2v_model):
    """Average Word2Vec embeddings; returns shape (1, dim)."""
    dim = w2v_model.vector_size
    vec = np.zeros(dim, dtype='float32')
    n = 0
    for t in tokens:
        if t in w2v_model.wv:
            vec += w2v_model.wv[t]
            n += 1
    if n:
        vec /= n
    return vec.reshape(1, -1)


W2V = Word2Vec.load('word2vec_model.bin')
CLF = joblib.load('svm_model2.pkl')   # or rename your file to best_model.pkl and update this

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def reliablity():
    result = None
    if request.method == 'POST':
        user_text = request.form.get('text', '')
        tokens = preprocess_tokens(user_text)
        doc = to_doc_vec(tokens, W2V)
        pred = int(CLF.predict(doc)[0])
        result = 'Reliable' if pred == 1 else 'Unreliable'
    return render_template('index.html', prediction_text=(f'News Reliability Analysis: {result}' if result else ''))

if __name__ == '__main__':
    app.run(debug=True)
