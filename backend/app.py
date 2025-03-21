import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import cossim as cos

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")


@app.route("/search")
def search():
    query = request.args.get("query")
    docs = data
    tokenized_docs = [{'toks': cos.tokenize(doc['abstract'])} for doc in docs]
    inv_index = cos.build_inverted_index(tokenized_docs)
    result = cos.search(query, docs, inv_index)
    return Flask.jsonify(result)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)