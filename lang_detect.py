import re
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


naive_model = pickle.load(open('naive_lang_detect_model1.pkl','rb'))

df = pd.read_csv('Language Detection.csv')

x = df.Text
y = df.Language

le = LabelEncoder()
encoded_y = le.fit_transform(y)

data = []
for text in x:
    text = re.sub(r'[!@#$(),.n"%^*?:;~`0-9\n]', '', text)
    text = text.lower()
    data.append(text)

cv = CountVectorizer()
x_vector = cv.fit_transform(data).toarray()

def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

    
"""
encode = {
3:'English',  
4: 'French', 
13: 'Spanish',
11: 'Portugeese',
8: 'Italian',
12 : 'Russian',
14: 'Sweedish',
10: 'Malayalam',
2 : 'Dutch',
0 : 'Arabic',
16 : 'Turkish',
5 : 'German',
15 : 'Tamil',
1 : 'Danish',
9 : 'Kannada',
6 : 'Greek',
7 : 'Hindi',
}
"""
