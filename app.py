import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.data.path.append('nltk_data')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize

with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message below to classify it as **Spam** or **Ham** (Not Spam).")

user_input = st.text_area("Enter SMS Message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        processed_input = preprocess_text(user_input)
        vect_input = vectorizer.transform([processed_input])
        prediction = model.predict(vect_input)[0]
        if prediction == 1:
            st.error("📛 This message is **Spam**.")
        else:
            st.success("✅ This message is **Ham** (Not Spam).")
