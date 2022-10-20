import pickle 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


naive_model = pickle.load(open('naive_lang_detect_model1.pkl','rb'))

df = pd.read_csv('Language Detection.csv')

x = df.Text
y = df.Language

le = LabelEncoder()
encoded_y = le.fit_transform(y)
data = pickle.load(open('data.pkl','rb'))
cv = CountVectorizer()
x_vector = cv.fit_transform(data).toarray()

def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

