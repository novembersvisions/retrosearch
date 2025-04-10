import json
import os
import time
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import cossim as cos
from paper_similarity import SimilarityMatrix
import random

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
    data = []

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
    inv_index = {}

# Initialize similarity matrix with caching
try:
    similarity_matrix = SimilarityMatrix(json_file_path)
    paper_similarities = similarity_matrix.load_or_compute(data)
    print(f"Successfully loaded/computed similarities for {len(data)} papers")
except Exception as e:
    print(f"Error with similarity matrix: {str(e)}")
    paper_similarities = []

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
        result = []
    
    return jsonify(result)

@app.route("/map")
def paper_map():
    """Render the paper map visualization page"""
    return render_template('map.html', title="ReSearch Paper Map")

@app.route("/map_data")
def map_data():
    """API endpoint for paper map visualization data"""
    query = request.args.get("query", "")
    print(f"Received map data query: '{query}'")
    
    if not query:
        return jsonify({"center": None, "related": []})
    
    try:
        start_time = time.time()
        
        # Get search results using existing search function
        results = cos.search(query, data, inv_index)
        
        if not results or len(results) == 0:
            return jsonify({"center": None, "related": []})
        
        # Main paper is the top search result
        center_paper = results[0]
        
        # Find the index of this paper in the original data
        center_index = None
        for i, paper in enumerate(data):
            if paper.get("title") == center_paper.get("title"):
                center_index = i
                break
        
        # Get related papers from precomputed similarities
        related_papers = []
        if center_index is not None and center_index < len(paper_similarities):
            related_papers = paper_similarities[center_index]
        else:
            # Fallback: use next top search results
            for i, paper in enumerate(results[1:16], 1):  # Limit to 15 related papers
                related_papers.append({
                    "id": i,
                    "original_id": -1,  # No original ID for search results
                    "title": paper.get("title", "Unknown"),
                    "abstract": paper.get("abstract", ""),
                    "link": paper.get("link", ""),
                    "score": max(0.3, 1.0 - (i * 0.05))  # Decreasing score
                })
        
        # Format data for the visualization
        map_data = {
            "center": {
                "id": 0,
                "original_id": center_index,
                "title": center_paper.get("title", "Unknown"),
                "abstract": center_paper.get("abstract", ""),
                "link": center_paper.get("link", ""),
                "score": 1.0  # Main paper has highest score
            },
            "related": related_papers
        }
        
        end_time = time.time()
        print(f"Map data generated in {end_time - start_time:.2f} seconds")
        
        return jsonify(map_data)
    except Exception as e:
        print(f"Error generating map data: {str(e)}")
        return jsonify({"center": None, "related": [], "error": str(e)})

@app.route("/paper_as_query")
def paper_as_query():
    """API endpoint to use a specific paper as the query center"""
    paper_id_str = request.args.get("paper_id", "")
    
    try:
        paper_id = int(paper_id_str)
        print(f"Using paper ID {paper_id} as query center")
        
        if paper_id < 0 or paper_id >= len(data):
            print(f"Invalid paper ID: {paper_id}")
            return jsonify({"center": None, "related": [], "error": "Invalid paper ID"})
        
        start_time = time.time()
        
        # Get the center paper data
        center_paper = data[paper_id]
        
        # Get related papers directly from the precomputed similarity matrix
        related_papers = []
        if paper_id < len(paper_similarities):
            # Get the similarity data for this paper
            similar_papers = paper_similarities[paper_id]
            
            for i, similar_paper in enumerate(similar_papers):
                # Include the original ID for reference
                related_papers.append({
                    "id": i + 1,  # ID for visualization (1-based)
                    "original_id": similar_paper.get("original_id"),
                    "title": similar_paper.get("title"),
                    "abstract": similar_paper.get("abstract"),
                    "link": similar_paper.get("link"),
                    "score": similar_paper.get("score")
                })
        
        # Format data for the visualization
        map_data = {
            "center": {
                "id": 0,
                "original_id": paper_id,
                "title": center_paper.get("title", "Unknown"),
                "abstract": center_paper.get("abstract", ""),
                "link": center_paper.get("link", ""),
                "score": 1.0  # Main paper has highest score
            },
            "related": related_papers
        }
        
        end_time = time.time()
        print(f"Paper-as-query map data generated in {end_time - start_time:.2f} seconds")
        
        return jsonify(map_data)
    except ValueError:
        print(f"Invalid paper ID format: {paper_id_str}")
        return jsonify({"center": None, "related": [], "error": "Invalid paper ID format"})
    except Exception as e:
        print(f"Error generating paper-as-query map: {str(e)}")
        return jsonify({"center": None, "related": [], "error": str(e)})

@app.route("/team")
def team():
    return render_template('team.html', title="ReSearch Team")
    
@app.route("/explore")
def explore():
    """Render the research explore visualization page"""
    return render_template('explore.html', title="ReSearch Galaxy Explorer")

@app.route("/explore_data")
def explore_data():
    """API endpoint for explore visualization data"""
    try:
        start_time = time.time()
        
        # Get a sample of papers to display
        sample_size = min(150, len(data))  # Limit to 150 papers for performance
        sampled_papers = data[:sample_size]
        
        # Prepare paper data for visualization
        papers = []
        for i, paper in enumerate(sampled_papers):
            # Check for required fields and set defaults if missing
            title = paper.get("title", "Untitled Paper")
            abstract = paper.get("abstract", "No abstract available")
            link = paper.get("link", "#")
            
            # Create a paper object with the fields needed by the visualization
            paper_obj = {
                "id": f"paper-{i}",
                "title": title,
                "abstract": abstract,
                "authors": paper.get("authors", "Unknown Author"),
                "year": paper.get("year", 2023),
                "citations": paper.get("citations", random.randint(0, 2000)),  # Random citation count if not present
                "link": link,
                "cluster": determine_cluster(title, abstract),
                "readingLevel": determine_reading_level(abstract),
                "related": []  # Will be populated with related paper IDs
            }
            
            papers.append(paper_obj)
        
        # Add related papers based on similarity
        for i, paper in enumerate(papers):
            if i < len(paper_similarities):
                # Get similar papers
                similar_papers = paper_similarities[i][:5]  # Limit to top 5 for simplicity
                related_ids = []
                
                for similar in similar_papers:
                    similar_id = similar.get("original_id")
                    if similar_id is not None and similar_id < len(papers):
                        related_ids.append(f"paper-{similar_id}")
                
                paper["related"] = related_ids
            else:
                # If no similarities, add some random connections
                count = min(3, len(papers) - 1)
                random_indices = random.sample([j for j in range(len(papers)) if j != i], count)
                paper["related"] = [f"paper-{j}" for j in random_indices]
        
        # Create guided journeys
        journeys = create_guided_journeys(papers)
        
        # Format data for the visualization
        explore_data = {
            "papers": papers,
            "journeys": journeys
        }
        
        end_time = time.time()
        print(f"Explore data generated in {end_time - start_time:.2f} seconds")
        
        return jsonify(explore_data)
    except Exception as e:
        print(f"Error generating explore data: {str(e)}")
        return jsonify({"papers": [], "journeys": [], "error": str(e)})

def determine_cluster(title, abstract):
    """Determine the research domain/cluster for a paper based on its title and abstract"""
    combined_text = (title + " " + abstract).lower()
    
    # Specific model names detection
    if "transformer" in combined_text or "attention is all you need" in combined_text:
        return "NLP"
    elif "elmo" in combined_text or "deep contextualized word" in combined_text:
        return "NLP"
    elif "xlnet" in combined_text or "bert" in combined_text or "gpt" in combined_text:
        return "NLP"
    
    # General domain detection
    if "machine learning" in combined_text or "neural network" in combined_text:
        return "Machine Learning"
    elif "computer vision" in combined_text or "image" in combined_text:
        return "Computer Vision"
    elif "natural language" in combined_text or "nlp" in combined_text or "language model" in combined_text:
        return "NLP"
    elif "reinforcement" in combined_text or "reward" in combined_text:
        return "Reinforcement Learning"
    elif "generative" in combined_text or "gan" in combined_text or "diffusion" in combined_text:
        return "Generative Models"
    else:
        # Fallback to one of the domains randomly
        domains = ["Machine Learning", "Computer Vision", "NLP", "Reinforcement Learning", "Generative Models"]
        return domains[hash(title) % len(domains)]

def determine_reading_level(abstract):
    """Determine the reading complexity of a paper"""
    words = abstract.split()
    if not words:
        return "Introductory"
    
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    if avg_word_length > 7:
        return "Advanced"
    elif avg_word_length > 5.5:
        return "Intermediate"
    else:
        return "Introductory"

def create_guided_journeys(papers):
    """Create guided journeys through papers"""
    # Group papers by cluster
    clusters = {}
    for paper in papers:
        cluster = paper["cluster"]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(paper)
    
    journeys = []
    
    # Journey templates by research domain
    journey_descriptions = {
        "Machine Learning": {
            "title": "Introduction to ML Fundamentals",
            "description": "A curated path through essential machine learning concepts and algorithms for beginners.",
            "class": "ml"
        },
        "Computer Vision": {
            "title": "Computer Vision Breakthrough Papers",
            "description": "Explore the most influential papers that revolutionized computer vision techniques.",
            "class": "cv"
        },
        "NLP": {
            "title": "NLP Evolution: From RNNs to Transformers",
            "description": "Follow the evolution of natural language processing from recurrent networks to transformer architectures.",
            "class": "nlp"
        },
        "Reinforcement Learning": {
            "title": "Foundations of Reinforcement Learning",
            "description": "Understand the core principles and algorithms behind reinforcement learning.",
            "class": "rl"
        },
        "Generative Models": {
            "title": "Generative AI Revolution",
            "description": "Discover the papers that led to today's powerful generative AI systems.",
            "class": "gen"
        }
    }
    
    for i, (cluster, cluster_papers) in enumerate(clusters.items()):
        # Skip if not enough papers
        if len(cluster_papers) < 3:
            continue
        
        journey_info = journey_descriptions.get(cluster, {
            "title": f"Journey through {cluster}",
            "description": f"Explore important papers in {cluster}.",
            "class": "default"
        })
        
        # Sort papers by citations (or randomly if no citations) and take a subset
        sorted_papers = sorted(cluster_papers, key=lambda p: p.get("citations", 0), reverse=True)
        journey_papers = sorted_papers[:min(5, len(sorted_papers))]  # Take top 5 papers or fewer
        
        journeys.append({
            "id": f"journey-{i+1}",
            "title": journey_info["title"],
            "description": journey_info["description"],
            "domain": cluster,
            "steps": [p["id"] for p in journey_papers],
            "class": journey_info["class"]
        })
    
    return journeys

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)