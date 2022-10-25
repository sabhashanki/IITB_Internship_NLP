import pickle 
import sys
sys.path.append('pickle_exports')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level = logging.INFO, filename = 'lang_detect_python.log', filemode = 'w', format = '%(asctime)s - %(levelname)s - %(message)s')

# Initialization
naive_model = pickle.load(open('pickle_exports/naive_lang_detect_model.pkl','rb'))
data = pickle.load(open('pickle_exports/data.pkl','rb'))
y = pickle.load(open('pickle_exports/y.pkl','rb'))
le = LabelEncoder()
cv = CountVectorizer()

#label encoding and vectorization
encoded_y = le.fit_transform(y)
x_vector = cv.fit_transform(data).toarray()
logging.info('encoded y and data vectorized')

#Prediction module 
def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    logging.info('prediction function executed')
    return lang[0]
    
# print(lang_prediction('hello world'))