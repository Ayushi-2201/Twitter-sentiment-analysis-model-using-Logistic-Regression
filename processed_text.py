import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# --- Downloading NLTK Data ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
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