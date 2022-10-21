from flask import Flask, render_template, request, jsonify
from key_extract import extract, hashtagg
from lang_detect import lang_prediction
from topic_model import topic_prediction


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
"""
@app.route('/predict', methods = ['POST'])
def ReturnJSON():
    data = request.form['data']
    keywords = extract(data)
    Keyword1, Keyword2, Keyword3, Keyword4, Keyword5 = keywords[0], keywords[1], keywords[2], keywords[3], keywords[4]
    lang = lang_prediction(data)
    topic = topic_prediction(data)
    if(request.method == 'POST'):
        data = {
        "Language": {"lang_code":lang,"conf_score":"value"},
        "Keywords": [ 
                    {"Keyword1":Keyword1, "conf_score":"value"}, {"Keyword2":Keyword2,"conf_score":"value"}, {"Keyword3":Keyword3,"conf_score":"value"}, {"Keyword4":Keyword4,"conf_score":"value"}, {"Keyword5":Keyword5,"conf_score":"value"}
                    ],
        "Topic": {"topic_name":topic,"conf_score":"value"},
        "Hashtags": [
                    {"hashtag1":"value","conf_score":"value"}, {"hashtag1":"value","conf_score":"value"}, {"hashtag1":"value","conf_score":"value"}, {"hashtag1":"value","conf_score":"value"}, {"hashtag1":"value","conf_score":"value"}
                    ]
                }
    return jsonify(data)
"""
# Driver Code
if __name__ == '__main__':
    app.run(debug = True)