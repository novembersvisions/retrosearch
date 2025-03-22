import json
import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import cossim as cos

# ROOT_PATH for linking with all your files. 
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

try:
    print(f"Loading JSON from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"Successfully loaded JSON with {len(data)} entries")
    
    # Print a sample entry to verify structure
    if data:
        print("First entry keys:", list(data[0].keys()))
except Exception as e:
    print(f"Error loading JSON: {str(e)}")

# Process the tokenization
try:
    for doc in data:
        doc['toks'] = cos.tokenize(doc["abstract"])
    print("Tokenization successful")
except Exception as e:
    print(f"Error during tokenization: {str(e)}")

# Build inverted index
try:
    inv_index = cos.build_inverted_index(data)
    print("Successfully built inverted index")
except Exception as e:
    print(f"Error building inverted index: {str(e)}")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html', title="ReSearch")

@app.route("/search")
def search():
    query = request.args.get("query", "")
    print(f"Received search query: '{query}'")
    
    if not query:
        return jsonify([])
    
    try:
        result = cos.search(query, data, inv_index)
        print(f"Search returned {len(result)} results")
        if result and len(result) > 0:
            print("First result keys:", list(result[0].keys()))
    except Exception as e:
        print(f"Error during search: {str(e)}")
    
    return jsonify(result)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)