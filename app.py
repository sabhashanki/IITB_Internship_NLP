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

# Logging
logging.basicConfig(level = logging.INFO, filename = 'logs/app.log', filemode = 'w', format = '%(asctime)s - %(levelname)s - %(message)s')
logging.info('All libraries exported')

# Initialization
punch = set(string.punctuation)
allstopwords = stopwords.words('english')
le = LabelEncoder()
cv = CountVectorizer()
lemma = WordNetLemmatizer()
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
extract_ngram = (1,1)
hashtag_ngram = (1,2)
top_n = 5

try:
    naive_model = pickle.load(open('pickle_exports/naive_lang_detect_model.pkl','rb'))
    data = pickle.load(open('pickle_exports/data.pkl','rb'))
    y = pickle.load(open('pickle_exports/y.pkl','rb'))
    logging.info('Pickle import successful')
except:
    logging.error('Pickle import failed')


# Label Encoding and Vectorization
try:
    encoded_y = le.fit_transform(y)
    x_vector = cv.fit_transform(data).toarray()
    logging.info('encoded y and data vectorized')
except:
    logging.error('Encode and vectorization failed')


# Data Cleaning Module
def clean(data):
    stop_free = " ".join([i for i in data.lower().split() if i not in allstopwords])    #removing stopwords 
    punc_free = ''.join(ch for ch in stop_free if ch not in punch)                      #removing punctuations
    data = " ".join(lemma.lemmatize(word) for word in punc_free.split())                #lemmatizing
    data = " ".join(re.sub(r'\d', "", word) for word in data.split())                   #removing empty spaces
    logging.info('Data cleaning successful')
    return data

# Keyword Extraction Module
def extract(data):
    data = clean(data)
    cvector = CountVectorizer(ngram_range=extract_ngram)                                        
    cvector.fit_transform([data])
    keywords = cvector.get_feature_names_out()
    data_embed = model.encode([data])
    keyword_embed = model.encode(keywords)
    distances = cosine_similarity(data_embed, keyword_embed)
    final_keywords = [keywords[index] for index in distances.argsort()[0][-top_n:]]
    logging.info('Keyword extraction module executed succussfully')
    return final_keywords

# Hashtag Prediction Module
def hashtagg(data):
    data = clean(data)
    cvector = CountVectorizer(ngram_range=hashtag_ngram)
    cvector.fit_transform([data])
    keywords = cvector.get_feature_names_out()
    data_embed = model.encode([data])
    keyword_embed = model.encode(keywords)
    distances = cosine_similarity(data_embed, keyword_embed)
    final_keywords = [keywords[index] for index in distances.argsort()[0][-top_n:]]
    final_keywords = ['#' + word.replace(' ','') for word in final_keywords]
    logging.info('Hashtag module executed')
    return final_keywords

# Topic Prediction Module
def topic_prediction(data):
    cleaned_data = [clean(line).split() for line in [data]]    
    dictionary = corpora.Dictionary(cleaned_data)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_data]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary, passes=100)
    logging.info('Topic prediction module executed')
    return (ldamodel.print_topics(num_topics=1, num_words=1)[0][1])

# Prediction Module 
def lang_prediction(text):
    x = cv.transform([text]).toarray()
    lang = naive_model.predict(x)
    lang = le.inverse_transform(lang)
    logging.info('prediction function executed')
    return lang[0]

# Flask API
app = Flask(__name__)

# Home Page
@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

# Predict Page
@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form['data']
    keywords = extract(data)
    lang = lang_prediction(data)
    topic = [i.split('*')[1] for i in list([topic_prediction(data)])]
    hashta = hashtagg(data)
    logging.info('Flask prediction module executed')
    return render_template('home.html', prediction_text1 = f'Language : {lang}', prediction_text2 = f'Topic prediction : {topic}', prediction_text3 = f'Important keywords : {keywords}', prediction_text4 = f'Predicted Hashtags : {hashta}')

# JSON Prediction Page
@app.route('/json_predict', methods = ['POST'])
def ReturnJSON():
    data = request.form['data']
    keywords = extract(data)
    lang = lang_prediction(data)
    topic = [i.split('*')[1] for i in list([topic_prediction(data)])]
    hashtags = hashtagg(data)
    if(request.method == 'POST'):
        data = {
        
        "Hashtags": [
                    {"hashtag1":hashtags[0],"conf_score":"value"}, 
                    {"hashtag2":hashtags[1],"conf_score":"value"}, 
                    {"hashtag3":hashtags[2],"conf_score":"value"}, 
                    {"hashtag4":keywords[3],"conf_score":"value"}, 
                    {"hashtag5":keywords[4],"conf_score":"value"}
        ],
        "Keywords": [
                    {"Keyword1":keywords[0], "conf_score":"value"}, 
                    {"Keyword2":keywords[1],"conf_score":"value"}, 
                    {"Keyword3":keywords[2],"conf_score":"value"}, 
                    {"Keyword4":keywords[3],"conf_score":"value"}, 
                    {"Keyword5":keywords[4],"conf_score":"value"}
        ],        

        "Topic": {"topic_name":topic,"conf_score":"value"},        
        "Language": {"lang_code":lang,"conf_score":"value"},
                
        }
    return jsonify(data)

# Driver Code
if __name__ == '__main__':
    app.run(debug = True)