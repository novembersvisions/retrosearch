# paper_similarity.py
import math
import json
import os
import time
from collections import Counter
import hashlib

class SimilarityMatrix:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.matrix_path = None
        self.data_hash = None
        self.similarity_matrix = []
        
        if data_path:
            self.matrix_path = os.path.join(os.path.dirname(data_path), 'similarity_matrix.json')
            
    def compute_data_hash(self, data):
        """Compute hash of the data to check if it has changed"""
        # We'll hash just the titles and abstracts since that's what affects similarity
        hash_data = []
        for item in data:
            hash_data.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstract", "")
            })
        
        data_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def load_or_compute(self, data):
        """Load existing similarity matrix if it exists and is valid, otherwise compute it"""
        current_hash = self.compute_data_hash(data)
        
        # Check if we have a cached matrix
        if self.matrix_path and os.path.exists(self.matrix_path):
            try:
                with open(self.matrix_path, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if the hash matches
                if cached_data.get('data_hash') == current_hash:
                    print(f"Loading similarity matrix from cache: {self.matrix_path}")
                    self.similarity_matrix = cached_data.get('matrix', [])
                    self.data_hash = current_hash
                    return self.similarity_matrix
                else:
                    print("Data has changed, recomputing similarity matrix...")
            except Exception as e:
                print(f"Error loading cached similarity matrix: {str(e)}")
        
        # Compute new matrix
        print("Computing new similarity matrix...")
        start_time = time.time()
        self.similarity_matrix = self.precompute_paper_similarities(data)
        self.data_hash = current_hash
        end_time = time.time()
        print(f"Similarity matrix computation took {end_time - start_time:.2f} seconds")
        
        # Save to cache
        if self.matrix_path:
            try:
                with open(self.matrix_path, 'w') as f:
                    json.dump({
                        'data_hash': self.data_hash,
                        'matrix': self.similarity_matrix
                    }, f)
                print(f"Saved similarity matrix to cache: {self.matrix_path}")
            except Exception as e:
                print(f"Error saving similarity matrix: {str(e)}")
        
        return self.similarity_matrix
    
    def compute_tf(self, text_tokens):
        """Compute term frequency for the tokens of a text"""
        if not text_tokens:
            return {}
        
        tf_dict = Counter(text_tokens)
        # Normalize by document length
        for term in tf_dict:
            tf_dict[term] = tf_dict[term] / len(text_tokens)
        
        return tf_dict

    def compute_idf(self, corpus_tokens_list):
        """Compute inverse document frequency for all terms in corpus"""
        num_docs = len(corpus_tokens_list)
        idf_dict = {}
        
        # Count documents containing each term
        term_doc_count = Counter()
        for doc_tokens in corpus_tokens_list:
            # Count each term only once per document
            for term in set(doc_tokens):
                term_doc_count[term] += 1
        
        # Calculate IDF for each term
        for term, doc_count in term_doc_count.items():
            idf_dict[term] = math.log(num_docs / (1 + doc_count))
        
        return idf_dict

    def compute_tfidf_vectors(self, corpus_tokens_list):
        """Compute TF-IDF vectors for all documents in corpus"""
        # Compute IDF values for all terms
        idf_dict = self.compute_idf(corpus_tokens_list)
        
        # Compute TF-IDF vectors for each document
        tfidf_vectors = []
        for doc_tokens in corpus_tokens_list:
            tf_dict = self.compute_tf(doc_tokens)
            tfidf_dict = {}
            
            for term, tf_value in tf_dict.items():
                tfidf_dict[term] = tf_value * idf_dict.get(term, 0)
            
            tfidf_vectors.append(tfidf_dict)
        
        return tfidf_vectors

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two sparse vectors (dictionaries)"""
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        # If no common terms, similarity is 0
        if not common_terms:
            return 0.0
        
        # Compute dot product for common terms
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        magnitude1 = math.sqrt(sum(val * val for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val * val for val in vec2.values()))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def precompute_paper_similarities(self, data):
        """Precompute similarity scores between all papers"""
        # Extract tokens from all papers
        all_tokens = [paper.get("toks", []) for paper in data]
        
        # Compute TF-IDF vectors for all papers
        tfidf_vectors = self.compute_tfidf_vectors(all_tokens)
        
        # Compute pairwise similarities
        similarity_matrix = []
        
        for i, paper1 in enumerate(data):
            similarities = []
            
            for j, paper2 in enumerate(data):
                if i == j:
                    # Skip comparing paper to itself
                    continue
                    
                # Compute similarity between papers using TF-IDF vectors
                sim_score = self.cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
                
                # Store the paper info with its similarity score
                similarities.append({
                    "id": j + 1,  # +1 because center will have id 0
                    "original_id": j,  # Store the original index for reference
                    "title": paper2.get("title", "Unknown"),
                    "abstract": paper2.get("abstract", ""),
                    "link": paper2.get("link", ""),
                    "score": sim_score
                })
            
            # Sort similarities in descending order
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # Store the top similar papers (limit to 15)
            similarity_matrix.append(similarities[:15])
        
        return similarity_matrix
        
    def get_similar_papers(self, paper_index, limit=15):
        """Get similar papers for a given paper index"""
        if paper_index < 0 or paper_index >= len(self.similarity_matrix):
            return []
            
        return self.similarity_matrix[paper_index][:limit]