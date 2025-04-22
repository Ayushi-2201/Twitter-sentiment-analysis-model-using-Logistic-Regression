import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
# ---------------------------------------------------------------

# --- Load the pre-trained models ---

# Use st.cache_resource for loading models/data to improve performance
@st.cache_resource
def load_vectorizer(path='tfidf_vectorizer.pkl'):
    """Loads the saved TF-IDF Vectorizer."""
    try:
        with open(path, 'rb') as file:
            vectorizer = pickle.load(file)
        return vectorizer
    except FileNotFoundError:
        st.error(f"Vectorizer file not found at {path}. Please ensure 'tfidf_vectorizer.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        return None

@st.cache_resource
def load_model(path='twitter_sentiment_analysis_model.sav'):
    """Loads the saved Logistic Regression Model."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure 'twitter_sentiment_analysis_model.sav' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the resources
vectorizer = load_vectorizer()
model = load_model()

# --- Preprocessing Function (copied from notebook) ---
port_stemmer = PorterStemmer()

def stemming(content):
  # removing every character that is not alphabetic in the content
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  # converting every character to lowercase
  stemmed_content = stemmed_content.lower()
  # splitting the content by space (" ") into a list
  stemmed_content = stemmed_content.split()
  # applying port_stemmer object method 'stem' to all the words (elements) in content
  # except for the stopwords that are present in the list stemmed_content.
  stemmed_content = [port_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  # joining the list to convert it into a string
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content
# --------------------------------------------------------

# --- Streamlit App Interface ---

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict whether it's positive or negative.")

# Input text area
user_input = st.text_area("Tweet Text:", height=100)

# Prediction button
if st.button("Predict Sentiment"):
    if user_input and vectorizer is not None and model is not None:
        # 1. Preprocess the input
        processed_input = stemming(user_input)

        # 2. Vectorize the processed input
        #    vectorizer.transform expects an iterable (list, etc.)
        vectorized_input = vectorizer.transform([processed_input])

        # 3. Predict using the loaded model
        prediction = model.predict(vectorized_input)

        # 4. Display the result
        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.success("Positive Tweet 😊")
            st.balloons()
        elif prediction[0] == 0:
            st.error("Negative Tweet 😞")

    elif not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        # Error messages for missing model/vectorizer are handled during loading
        st.info("Please ensure the model and vectorizer files are loaded correctly.")

st.write("---")