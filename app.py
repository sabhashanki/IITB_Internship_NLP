from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.preprocessing import LabelEncoder
import logging
import pickle 
import gensim
from gensim import corpora
from flask import Flask, render_template, request, jsonify


#assigning punctuations, stopwords, ngram and initializing lemmatizer
punch = set(string.punctuation)
allstopwords = stopwords.words('English')
lemma = WordNetLemmatizer()
extract_ngram = (1,1)
hashtag_ngram = (1,2)

# Initialization
naive_model = pickle.load(open('naive_lang_detect_model.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))
y = pickle.load(open('y.pkl','rb'))
le = LabelEncoder()
cv = CountVectorizer()

#label encoding and vectorization
encoded_y = le.fit_transform(y)
x_vector = cv.fit_transform(data).toarray()
logging.info('encoded y and data vectorized')

#cleaning the data
def clean(data):
    stop_free = " ".join([i for i in data.lower().split() if i not in allstopwords])    #removing stopwords 
    punc_free = ''.join(ch for ch in stop_free if ch not in punch)                      #removing punctuations
    data = " ".join(lemma.lemmatize(word) for word in punc_free.split())                #lemmatizing
    data = " ".join(re.sub(r'\d', "", word) for word in data.split())                   #removing empty spaces
    return data

def extract(data):
    data = clean(data)
    cvector = CountVectorizer(ngram_range=extract_ngram)                                        
    cvector.fit_transform([data])
    keywords = cvector.get_feature_names_out()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    data_embed = model.encode([data])
    keyword_embed = model.encode(keywords)
    global top_n
    top_n = 5
    distances = cosine_similarity(data_embed, keyword_embed)
    final_keywords = [keywords[index] for index in distances.argsort()[0][-top_n:]]
    return final_keywords

def hashtagg(data):
    data = clean(data)
    cvector = CountVectorizer(ngram_range=hashtag_ngram)
    cvector.fit_transform([data])
    keywords = cvector.get_feature_names_out()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    data_embed = model.encode([data])
    keyword_embed = model.encode(keywords)
    #top_n = 5
    distances = cosine_similarity(data_embed, keyword_embed)
    final_keywords = [keywords[index] for index in distances.argsort()[0][-top_n:]]
    final_keywords = ['#' + word.replace(' ','') for word in final_keywords]
    return final_keywords

def topic_prediction(data):
    cleaned_data = [clean(line).split() for line in [data]]    
    dictionary = corpora.Dictionary(cleaned_data)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_data]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary, passes=100)
    return (ldamodel.print_topics(num_topics=1, num_words=1)[0][1])

#Prediction module 
def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    logging.info('prediction function executed')
    return lang[0]


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form['data']
    keywords = extract(data)
    lang = lang_prediction(data)
    topic = [i.split('*')[1] for i in list([topic_prediction(data)])]
    hashta = hashtagg(data)
    return render_template('home.html', prediction_text1 = f'Text Language : {lang}', prediction_text2 = f'Text topic prediction is : {topic}', prediction_text3 = f'Important keywords : {keywords}', prediction_text4 = f'Predicted Hashtags : {hashta}')

# Driver Code
if __name__ == '__main__':
    app.run(debug = True)