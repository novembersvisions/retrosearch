import json
import os
import time
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import cossim as cos
from paper_similarity import SimilarityMatrix
import random
import numpy as np
import pickle
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy import sparse

def modified_sigmoid(x):
    """
    Modified sigmoid function with logarithmic transformation
    optimized for range tens–hundreds of thousands mapping to 0.5–0.9
    with better sensitivity in the lower ranges.
    """
    a = 0.2979
    b = -1.2902
    # avoid log(0)
    return 1 / (1 + np.exp(-(a * np.log(x + 1) + b)))

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
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify([])

    # 1) exact title match check
    exact = [doc for doc in data if doc.get("title", "").lower() == query.lower()]
    if exact:
        # return exact match at top
        doc = exact[0].copy()
        doc["score"] = 1.0
        return jsonify(convert_to_serializable([doc]))

    try:
        # transform query into TF-IDF
        query_tfidf = vectorizer.transform([query])
        if query_tfidf.nnz == 0:
            # fallback keyword scan
            lowq = query.lower()
            hits = []
            for doc in data:
                if lowq in doc.get("title", "").lower() or lowq in doc.get("abstract", "").lower():
                    hits.append(doc)
                    if len(hits) >= 5:
                        break
            return jsonify(convert_to_serializable(hits))

        # 2) get TF-IDF cosine scores
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # 3) narrow down to top K by TF-IDF
        K = 50
        top_idxs = np.argsort(tfidf_scores)[::-1][:K]

        # 4) compute SVD-based similarity on those K
        query_svd = svd.transform(query_tfidf)
        query_norm = normalize(query_svd)[0]  # shape (n_components,)
        candidates = document_vectors_normalized[top_idxs]  # (K, n_components)
        svd_scores = candidates.dot(query_norm)

        # 5) combine lexical + latent
        alpha = 0.8
        combined = alpha * tfidf_scores[top_idxs] + (1 - alpha) * svd_scores

        # 6) boost by citation count
        boost_factor = 0.2
        boosted = []
        for idx_pos, doc_idx in enumerate(top_idxs):
            base_score = combined[idx_pos]
            cite_cnt = data[doc_idx].get("citation_count", 0)
            boost = modified_sigmoid(cite_cnt)
            # final = base * (1 + boost_factor * (boost - 0.5))
            final_score = base_score * (1 + boost_factor * (boost - 0.5))
            boosted.append((doc_idx, final_score))

        # 7) pick top N results
        N = 7
        best = sorted(boosted, key=lambda x: x[1], reverse=True)[:N]

        # 8) build result list
        results = []
        for idx, score in best:
            doc = data[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        print(f"Search returned {len(results)} results")
    except Exception as e:
        print(f"Error during search: {e}")
        results = []

    return jsonify(convert_to_serializable(results))

@app.route("/map")
def paper_map():
    """Render the paper map visualization page"""
    return render_template('map.html', title="ReSearch Paper Map")

@app.route("/map_data")
def map_data():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"center": None, "related": []})
    
    try:
        # Transform query to TF-IDF
        query_tfidf = vectorizer.transform([query])
        if query_tfidf.sum() == 0:
            return jsonify({"center": None, "related": []})
        
        # Project to SVD space and normalize
        query_svd = svd.transform(query_tfidf)
        query_normalized = normalize(query_svd)
        
        # Compute similarities
        scores = query_normalized @ document_vectors_normalized.T
        top_indices = np.argsort(scores[0])[::-1][:5]
        results = [data[i] for i in top_indices]
        
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
            # Get the similarity data for this paper
            similar_papers = paper_similarities[center_index]
            
            # Add citation counts to existing related papers
            for similar_paper in similar_papers:
                similar_id = similar_paper.get("original_id")
                if similar_id is not None and similar_id >= 0 and similar_id < len(data):
                    # Add citation count from the original data
                    similar_paper["citations"] = data[similar_id].get("citation_count", 0)
            
            related_papers = similar_papers
            
        else:
            # Fallback: use next top search results
            for i, paper in enumerate(results[1:16], 1):  # Limit to 15 related papers
                related_papers.append({
                    "id": i,
                    "original_id": -1,  # No original ID for search results
                    "title": paper.get("title", "Unknown"),
                    "abstract": paper.get("abstract", ""),
                    "link": paper.get("link", ""),
                    "score": max(0.3, 1.0 - (i * 0.05)),  # Decreasing score
                    "citations": paper.get("citation_count", 0)  # Use current paper's citation count
                })
        
        # Format data for the visualization
        map_data = {
            "center": {
                "id": 0,
                "original_id": center_index,
                "title": center_paper.get("title", "Unknown"),
                "abstract": center_paper.get("abstract", ""),
                "link": center_paper.get("link", ""),
                "score": 1.0,  # Main paper has highest score
                "citations": center_paper.get("citation_count", 0)
            },
            "related": related_papers
        }
        
        # Convert any NumPy types to Python standard types for JSON serialization
        map_data = convert_to_serializable(map_data)
        
        return jsonify(map_data)
    except Exception as e:
        print(f"Error generating map data: {str(e)}")
        return jsonify({"center": None, "related": [], "error": str(e)})
    
def convert_to_serializable(obj):
    """
    Convert numpy types to Python standard types for JSON serialization
    """
    import numpy as np
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

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
                original_id = similar_paper.get("original_id")
                citation_count = 0
                
                # Get citation count from the original data
                if original_id is not None and original_id >= 0 and original_id < len(data):
                    citation_count = data[original_id].get("citation_count", 0)
                
                # Include the original ID for reference
                related_papers.append({
                    "id": i + 1,  # ID for visualization (1-based)
                    "original_id": original_id,
                    "title": similar_paper.get("title"),
                    "abstract": similar_paper.get("abstract"),
                    "link": similar_paper.get("link"),
                    "score": similar_paper.get("score"),
                    "citations": citation_count
                })
        
        # Format data for the visualization
        map_data = {
            "center": {
                "id": 0,
                "original_id": paper_id,
                "title": center_paper.get("title", "Unknown"),
                "abstract": center_paper.get("abstract", ""),
                "link": center_paper.get("link", ""),
                "score": 1.0,  # Main paper has highest score
                "citations": center_paper.get("citation_count", 0)
            },
            "related": related_papers
        }
        
        end_time = time.time()
        print(f"Paper-as-query map data generated in {end_time - start_time:.2f} seconds")
        
        map_data = convert_to_serializable(map_data)
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
        
        explore_data = convert_to_serializable(explore_data)
        return jsonify(explore_data)
    except Exception as e:
        print(f"Error generating explore data: {str(e)}")
        return jsonify({"papers": [], "journeys": [], "error": str(e)})
    
@app.route("/reinforce_map", methods=["POST"])
def reinforce_map():
    """API endpoint for Rocchio reinforcement of map data"""
    try:
        request_data = request.json
        
        # Extract data from request
        center_id = request_data.get("center_id")
        selected_ids = request_data.get("selected_ids", [])
        unselected_ids = request_data.get("unselected_ids", [])
        
        if center_id is None or not selected_ids:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get center paper vector
        if center_id < 0 or center_id >= len(document_vectors_normalized):
            return jsonify({"error": "Invalid center paper ID"}), 400
            
        center_vector = document_vectors_normalized[center_id]
        
        # Get selected paper vectors (relevant documents)
        relevant_vectors = []
        for paper_id in selected_ids:
            if 0 <= paper_id < len(document_vectors_normalized):
                relevant_vectors.append(document_vectors_normalized[paper_id])
        
        # Get unselected paper vectors (non-relevant documents)
        nonrelevant_vectors = []
        for paper_id in unselected_ids:
            if 0 <= paper_id < len(document_vectors_normalized):
                nonrelevant_vectors.append(document_vectors_normalized[paper_id])
        
        # Apply Rocchio algorithm
        # Convert lists to numpy arrays
        relevant_vectors_array = np.array(relevant_vectors)
        nonrelevant_vectors_array = np.array(nonrelevant_vectors)
        
        updated_query = rocchio_feedback(
            center_vector,
            relevant_vectors_array,
            nonrelevant_vectors_array,
            alpha=1.0,
            beta=0.75,
            gamma=0.15
        )
        
        print("Original query vector (first 5):", center_vector[:5])
        print("Updated query vector (first 5):", updated_query[:5])

        # Normalize the updated query vector
        updated_query_norm = normalize(updated_query.reshape(1, -1))[0]
        
        # Find papers similar to the updated query
        scores = updated_query_norm @ document_vectors_normalized.T

        print(f"Center vector: {center_vector[:5]}")
        print(f"Num relevant: {len(relevant_vectors)}")
        print(f"Num non-relevant: {len(nonrelevant_vectors)}")
        print(f"Updated query (first 5): {updated_query[:5]}")
        print(f"Normalized query (first 5): {updated_query_norm[:5]}")
        print(f"Scores shape: {scores.shape}")
        
        # Get top papers, excluding center and already selected papers
        exclude_ids = [center_id] + selected_ids
        scored_papers = [(i, score) for i, score in enumerate(scores) 
                          if i not in exclude_ids]
                          
        # Sort by similarity score
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Take top papers (limit to 15 - len(selected_papers))
        max_related = 15 - len(selected_ids)
        top_indices = [idx for idx, _ in scored_papers[:max_related]]
        
        # Prepare related papers data including selected papers
        related_papers = []
        
        # First add the selected papers
        for i, paper_id in enumerate(selected_ids, 1):
            if 0 <= paper_id < len(data):
                paper = data[paper_id]
                related_papers.append({
                    "id": i,
                    "original_id": paper_id,
                    "title": paper.get("title", "Unknown"),
                    "abstract": paper.get("abstract", ""),
                    "link": paper.get("link", ""),
                    "score": 1.0,  # Mark as fully relevant
                    "citations": paper.get("citation_count", 0)
                })
        
        # Then add newly found papers
        start_id = len(related_papers) + 1
        for i, paper_id in enumerate(top_indices, start_id):
            paper = data[paper_id]
            related_papers.append({
                "id": i,
                "original_id": paper_id,
                "title": paper.get("title", "Unknown"),
                "abstract": paper.get("abstract", ""),
                "link": paper.get("link", ""),
                "score": float(scores[paper_id]),
                "citations": paper.get("citation_count", 0)
            })
        
        # Format data for the visualization
        map_data = {
            "center": {
                "id": 0,
                "original_id": center_id,
                "title": data[center_id].get("title", "Unknown"),
                "abstract": data[center_id].get("abstract", ""),
                "link": data[center_id].get("link", ""),
                "score": 1.0,
                "citations": data[center_id].get("citation_count", 0)
            },
            "related": related_papers
        }
        
        # Convert any NumPy types to Python standard types for JSON serialization
        map_data = convert_to_serializable(map_data)
        
        return jsonify(map_data)
    except Exception as e:
        print(f"Error in reinforcement: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def rocchio_feedback(
    query: np.ndarray,
    relevant_docs: np.ndarray,
    nonrelevant_docs: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15
    ) -> np.ndarray:
        """
        Apply Rocchio relevance feedback (reinforcement) to update a query vector.
        
        Parameters
        ----------
        query : np.ndarray
            Original or current query vector (shape: [n_features]).
        relevant_docs : np.ndarray
            Array of relevant document vectors (shape: [n_rel, n_features]).
        nonrelevant_docs : np.ndarray
            Array of non-relevant document vectors (shape: [n_nonrel, n_features]).
        alpha : float
            Weight for the original query vector.
        beta : float
            Weight for relevant-document centroid.
        gamma : float
            Weight for non-relevant-document centroid.
        
        Returns
        -------
        np.ndarray
            Updated query vector (shape: [n_features]).
        """
        # Compute centroids (handle empty sets gracefully)
        rel_centroid = np.mean(relevant_docs, axis=0) if len(relevant_docs) > 0 else np.zeros_like(query)
        nonrel_centroid = np.mean(nonrelevant_docs, axis=0) if len(nonrelevant_docs) > 0 else np.zeros_like(query)
        
        # Rocchio update rule
        updated_query = alpha * query + beta * rel_centroid - gamma * nonrel_centroid
        
        return updated_query

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

# Create precomputed directory
precompute_dir = os.path.join(current_directory, 'precomputed')
os.makedirs(precompute_dir, exist_ok=True)

# File paths
vectorizer_path = os.path.join(precompute_dir, 'tfidf_vectorizer.pkl')
tfidf_matrix_path = os.path.join(precompute_dir, 'tfidf_matrix.npz')
svd_path = os.path.join(precompute_dir, 'svd_model.pkl')
document_vectors_path = os.path.join(precompute_dir, 'document_vectors.npy')
similarities_path = os.path.join(precompute_dir, 'similarities.pkl')
precompute_hash_path = os.path.join(precompute_dir, 'data_hash.txt')

# Compute data hash
data_hash = hashlib.md5(json.dumps([{'title': d['title'], 'abstract': d['abstract']} for d in data]).encode()).hexdigest()

recompute = False
if os.path.exists(precompute_hash_path):
    with open(precompute_hash_path, 'r') as f:
        saved_data_hash = f.read().strip()
    if saved_data_hash != data_hash:
        recompute = True
else:
    recompute = True

if not recompute and all(os.path.exists(p) for p in [vectorizer_path, tfidf_matrix_path, svd_path, document_vectors_path, similarities_path]):
    # Load precomputed data
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    with open(svd_path, 'rb') as f:
        svd = pickle.load(f)
    document_vectors_normalized = np.load(document_vectors_path)
    with open(similarities_path, 'rb') as f:
        paper_similarities = pickle.load(f)
    print("Loaded precomputed SVD components and similarities")
else:
    # Recompute everything
    print("Precomputing TF-IDF components and similarities...")
    texts = [f"{doc.get('title','')} {doc['abstract']}" for doc in data]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
     stop_words='english',
     max_df=0.7,
     min_df=1,           
     ngram_range=(1,2)     
     )
    
    texts = [f"{doc.get('title','')} {doc['abstract']}" for doc in data]
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
        sparse.save_npz(tfidf_matrix_path, tfidf_matrix)
        
    # Truncated SVD
    n_components = 100
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    document_vectors = svd.fit_transform(tfidf_matrix)
    document_vectors_normalized = normalize(document_vectors, axis=1)
    
    # Compute similarities using Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=16, metric='cosine', algorithm='brute')
    nn.fit(document_vectors_normalized)
    distances, indices = nn.kneighbors(document_vectors_normalized)
    
    paper_similarities = []
    for i in range(len(indices)):
        similar_indices = indices[i][1:]  # Exclude self
        similar_scores = 1 - distances[i][1:]
        similar_papers = [{
            "id": idx,
            "original_id": idx,
            "title": data[idx]['title'],
            "abstract": data[idx]['abstract'],
            "link": data[idx]['link'],
            "score": score
        } for idx, score in zip(similar_indices, similar_scores)][:15]
        paper_similarities.append(similar_papers)
    
    # Save precomputed data
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(svd_path, 'wb') as f:
        pickle.dump(svd, f)
    np.save(document_vectors_path, document_vectors_normalized)
    with open(similarities_path, 'wb') as f:
        pickle.dump(paper_similarities, f)
    with open(precompute_hash_path, 'w') as f:
        f.write(data_hash)
    print("Precomputation complete")

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)