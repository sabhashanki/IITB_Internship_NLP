from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
import string

#assigning punctuations, stopwords and initializing lemmatizer
punch = set(string.punctuation)
allstopwords = stopwords.words('English')
lemma = WordNetLemmatizer()

#cleaning the data
def clean(data):
    stop_free = " ".join([i for i in data.lower().split() if i not in allstopwords])    #removing stopwords 
    punc_free = ''.join(ch for ch in stop_free if ch not in punch)                      #removing punctuations
    data = " ".join(lemma.lemmatize(word) for word in punc_free.split())                #lemmatizing
    data = " ".join(re.sub(r'\d', "", word) for word in data.split())                   #removing empty spaces
    return data

def extract(data):
    data = clean(data)
    cvector = CountVectorizer(ngram_range=(1,1))                                        
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
    cvector = CountVectorizer(ngram_range=(1,2))
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

#hashtag('High school or senior high school is the education students receive in the final stage of secondary education in the United States. In the United States this lasts from approximately 14 to 19 years old in most cases. Most comparable to secondary schools, high schools generally deliver phase three of the ISCED model of education. High schools have subject-based classes. The name high school is applied in other countries, but no universal generalization can be made as to the age range, financial status, or ability level of the pupils accepted. In North America, most high schools include grades nine through twelve. Students attend them following graduation from middle school (also known as junior high school).[1]')

