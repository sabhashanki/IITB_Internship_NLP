from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
import string

stop = set(stopwords.words('english'))
punc = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in punc)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def topic_prediction(data):
    doc_clean = [clean(doc).split() for doc in [data]]    
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary, passes=100)
    return (ldamodel.print_topics(num_topics=1, num_words=1)[0][1])

#topic_prediction(data)