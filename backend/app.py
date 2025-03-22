import json
import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
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

for doc in data:
    doc['toks'] = cos.tokenize(doc["abstract"])

inv_index = cos.build_inverted_index(data)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    
    print(f"Searching for: {query}")
    
    result = cos.search(query, data, inv_index)
    return jsonify(result)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)