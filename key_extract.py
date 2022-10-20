from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

allstopwords = stopwords.words('English')

def extract(data):
    cvector = CountVectorizer(ngram_range=(1,1), stop_words=allstopwords)
    cvector.fit_transform([data])
    keywords = cvector.get_feature_names_out()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    data_embed = model.encode([data])
    keyword_embed = model.encode(keywords)
    top_n = 5
    distances = cosine_similarity(data_embed, keyword_embed)
    final_keywords = [keywords[index] for index in distances.argsort()[0][-top_n:]]
    return final_keywords

