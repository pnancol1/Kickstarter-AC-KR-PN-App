import os
from pickle import load
import streamlit as st
import sklearn
import regex as re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer



pitch = st.text_input("Enter Your Pitch", "Enter Pitch here")

model = load(open("src/tech_model.sav", "rb"))
vectorizer = load(open("src/tech_vectorizer.sav", "rb")) 


# if os.path.exists("svc_rbf_model.sav"):
#     st.write("Model Loaded")

# if os.path.exists("large vectorizer.sav"):
#     st.write("Vectorizer Loaded")

class_dict = {
    "0": "Failure",
    "1": "Success",
}



def preprocess_text(text):
    # Remove any character that is not a letter (a-z) or white space ( )
    text = re.sub(r'[^a-zA-Z]', " ", text)
    
    # Remove white spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)

    # Multiple white spaces into one
    text = re.sub(r'\s+', " ", text.lower())

    # Remove tags
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)

    return text.split()



download("wordnet")
lemmatizer = WordNetLemmatizer()

download("stopwords")
stop_words = stopwords.words("english")
stop_words.append('designed')
stop_words.append('kickstarter')
stop_words.append('name')

def lemmatize_text(words, lemmatizer = lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens


# st.write("Other stuff loaded")
if st.button("Predict"):
    # st.write("Running")
    processed_pitch = preprocess_text(pitch)
    # st.write("Processed")
    lemmatized_pitch = lemmatize_text(processed_pitch)
    # st.write("Lemmatized")
    tokens_list = lemmatized_pitch
    tokens_list = [tokens_list]
    tokens_list = [" ".join(tokens) for tokens in tokens_list]
    # vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.8, min_df = 5)
    X = vectorizer.transform(tokens_list).toarray()

    # st.write(str(X))
    prediction = str(model.predict(X)[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)