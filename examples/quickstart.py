"""
Quickstart: in-memory QdrantEmbeddingsDB — add vectors and query nearest neighbours.

Run:
    python examples/quickstart.py
"""
import numpy as np
from ovos_qdrant_embeddings import QdrantEmbeddingsDB

# In-memory DB — no server or file path needed
db = QdrantEmbeddingsDB(config={"vector_size": 4})

# Store a few small vectors with metadata
db.add_embeddings("apple",  np.array([1.0, 0.0, 0.0, 0.0]), metadata={"category": "fruit"})
db.add_embeddings("banana", np.array([0.0, 1.0, 0.0, 0.0]), metadata={"category": "fruit"})
db.add_embeddings("cherry", np.array([0.0, 0.0, 1.0, 0.0]), metadata={"category": "fruit"})
db.add_embeddings("desk",   np.array([0.0, 0.0, 0.0, 1.0]), metadata={"category": "furniture"})

print(f"Stored {db.count_embeddings_in_collection()} vectors")

# Query: which vectors are closest to [1, 0.1, 0, 0]?
query = np.array([1.0, 0.1, 0.0, 0.0])
results = db.query(query, top_k=3, return_metadata=True)

print("\nNearest neighbours:")
for key, score, meta in results:
    print(f"  {key:8s}  score={score:.4f}  {meta}")

# Retrieve a specific vector
emb = db.get_embeddings("apple")
print(f"\nRetrieved 'apple' vector: {emb}")

# Delete one entry and verify it is gone
db.delete_embeddings("desk")
assert db.get_embeddings("desk") is None
print("\n'desk' deleted successfully")
