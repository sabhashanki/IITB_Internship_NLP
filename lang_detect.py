import pickle 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
logging.basicConfig(level = logging.INFO, filename = 'lang_detect_python.log', filemode = 'w', format = '%(asctime)s - %(levelname)s - %(message)s')

# Initialization
naive_model = pickle.load(open('naive_lang_detect_model1.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))
le = LabelEncoder()
cv = CountVectorizer()

# exporting data 
try:
    df = pd.read_csv('Language Detection.csv')
    logging.info('CSV file imported')
except:
    print('CSV file not found')
    logging.error('CSV file not found')


#label encoding and vectorization
y = df.Language
encoded_y = le.fit_transform(y)
x_vector = cv.fit_transform(data).toarray()

#Prediction module 
def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]
