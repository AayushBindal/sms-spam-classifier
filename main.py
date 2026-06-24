
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum(): y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam CLassifier")
input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. probability
    probs = model.predict_proba(vector_input)[0]

    ham_prob = probs[0] * 100
    spam_prob = probs[1] * 100

    # 5. Display result
    if result == 1:
        st.error("Spam Message")
        st.write(f"**Spam Confidence:** {spam_prob:.2f}%")
    else:
        st.success("Not Spam")
        st.write(f"**Ham Confidence:** {ham_prob:.2f}%")

    # 6. Show probabilities
    st.subheader("Prediction Probabilities")
    st.progress(float(max(ham_prob, spam_prob) / 100))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Ham", f"{ham_prob:.2f}%")

    with col2:
        st.metric("Spam", f"{spam_prob:.2f}%")
