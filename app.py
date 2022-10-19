from flask import Flask, render_template, request



app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():

           
    return render_template('home.html')

# Driver Code
if __name__ == '__main__':
    app.run(debug = True)