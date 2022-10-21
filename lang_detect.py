import pickle 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

#importing trained model    
naive_model = pickle.load(open('naive_lang_detect_model1.pkl','rb'))

#exporting csv file to retrieve y data
df = pd.read_csv('Language Detection.csv')
y = df.Language

#label encoder for y
le = LabelEncoder()
encoded_y = le.fit_transform(y)

#importing cleaned training data
data = pickle.load(open('data.pkl','rb'))

#vectorizing cleaned data
cv = CountVectorizer()
x_vector = cv.fit_transform(data).toarray()

#Prediction module 
def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

