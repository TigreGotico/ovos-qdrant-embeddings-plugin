"""
Remote Qdrant server example.

Requires a running Qdrant instance. Start one locally with Docker:

    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Or point at Qdrant Cloud by setting QDRANT_HOST and QDRANT_API_KEY.

Run:
    QDRANT_HOST=localhost python examples/remote_server.py
"""
import os
import numpy as np
from ovos_qdrant_embeddings import QdrantEmbeddingsDB

host = os.environ.get("QDRANT_HOST", "localhost")
api_key = os.environ.get("QDRANT_API_KEY")

config = {
    "host": host,
    "port": int(os.environ.get("QDRANT_PORT", 6333)),
    "grpc_port": int(os.environ.get("QDRANT_GRPC_PORT", 6334)),
    "vector_size": 4,
    "distance_metric": "cosine",
    "default_collection_name": "ovos_demo",
}
if api_key:
    config["api_key"] = api_key

print(f"Connecting to Qdrant at {host}:{config['port']}")
db = QdrantEmbeddingsDB(config=config)

# Add a few vectors
db.add_embeddings("remote-vec-1", np.array([1.0, 0.0, 0.0, 0.0]), metadata={"source": "demo"})
db.add_embeddings("remote-vec-2", np.array([0.0, 1.0, 0.0, 0.0]), metadata={"source": "demo"})

print(f"Stored {db.count_embeddings_in_collection()} vectors in '{db.default_collection_name}'")

# Query
results = db.query(np.array([0.9, 0.1, 0.0, 0.0]), top_k=2)
print("Query results:", results)

# Clean up
db.delete_embeddings("remote-vec-1")
db.delete_embeddings("remote-vec-2")
print("Cleaned up demo vectors")
