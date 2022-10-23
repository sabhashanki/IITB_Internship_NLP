from flask import Flask, render_template, request, jsonify
import sys
sys.path.append('Modules')
from key_extract_hashtag import extract, hashtagg
from lang_detect import lang_prediction
from Modules.topic_model import topic_prediction


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

# @app.route('/predict', methods = ['POST'])
# def predict():
#     data = request.form['data']
#     keywords = extract(data)
#     lang = lang_prediction(data)
#     topic = [i.split('*')[1] for i in list([topic_prediction(data)])]
#     hashta = hashtagg(data)
#     return render_template('home.html', prediction_text1 = f'Text Language : {lang}', prediction_text2 = f'Text topic prediction is : {topic}', prediction_text3 = f'Important keywords : {keywords}', prediction_text4 = f'Predicted Hashtags : {hashta}')

@app.route('/predict', methods = ['POST'])
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