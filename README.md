# IITB Internship NLP

### Report
	This project regards in identifying the language, extract important keywords, predict topic and hashtags using Natural Language Processing techniques. All modules are coded separately in both jupyter notebook and python file. Testing module written and tested for each module. Flask API’s created both in JSON and normal format which cal be accessed through client application. Pipelined CI/CD using Git action and Heroku cloud.

### References
    • Analytics Vidya - Topic Modelling
    • Topic Modelling Techniques
    • Topic Modelling Guide
    • Towards Science - Keyword Extraction
    • Keyword Extraction Guide
    • Keyword Extraction - BERT
    • Language Detection - Analytics Vidya

### Setup
    • Clone the repository 
    • Run jupyter_notebook/lang_detect.ipynb to export required pickle files
    • Testing files are in modules_testing folder
    • Logs - /logs 
    • Dataset - /dataset
    • Pickle files - /pickle_exports
      
### Client Script
    • Run app.py and access the html page through http://127.0.0.1:5000/
    • /predict – displays output within same page
    • /json_predict – displays JSON output in different page