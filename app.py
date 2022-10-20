from flask import Flask, render_template, request, jsonify
from key_extract import extract
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
    topic = topic_prediction(data)
    newline = '\n' * 2
    return render_template('home.html', prediction_text1 = f'Text Language : {lang}', prediction_text2 = f'Text topic prediction is : {topic}', prediction_text3 = f'Important keywords : {keywords}')

@app.route('/returnjson', methods = ['GET'])
def ReturnJSON():
    if(request.method == 'GET'):
        data = {
            "Modules" : 15,
            "Subject" : "Data Structures and Algorithms",
        }
  
        return jsonify(data)

# Driver Code
if __name__ == '__main__':
    app.run(debug = True)