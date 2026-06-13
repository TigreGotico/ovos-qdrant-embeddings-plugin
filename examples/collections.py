"""
Multi-collection example: separate vector spaces for different domains.

Run:
    python examples/collections.py
"""
import numpy as np
from ovos_qdrant_embeddings import QdrantEmbeddingsDB

db = QdrantEmbeddingsDB(config={
    "vector_size": 4,
    "default_collection_name": "utterances",
})

# Utterances collection (default)
db.add_embeddings("turn-on-lights", np.array([1.0, 0.0, 0.5, 0.0]), metadata={"intent": "lights.on"})
db.add_embeddings("play-music",     np.array([0.0, 1.0, 0.0, 0.5]), metadata={"intent": "media.play"})

# Separate collection for skill memories
db.create_collection("memories")
db.add_embeddings("reminder-1", np.array([0.3, 0.3, 0.8, 0.1]), metadata={"type": "reminder"}, collection_name="memories")
db.add_embeddings("note-1",     np.array([0.1, 0.8, 0.1, 0.3]), metadata={"type": "note"},     collection_name="memories")

print("Collections:", [c.name for c in db.list_collections()])
print("Utterances count:", db.count_embeddings_in_collection("utterances"))
print("Memories count:",   db.count_embeddings_in_collection("memories"))

# Query each collection independently
query = np.array([0.9, 0.1, 0.4, 0.0])

print("\nTop utterance match:")
for key, score in db.query(query, top_k=1, collection_name="utterances"):
    print(f"  {key}  score={score:.4f}")

print("\nTop memory match:")
for key, score in db.query(query, top_k=1, collection_name="memories"):
    print(f"  {key}  score={score:.4f}")

# Batch add to utterances
db.add_embeddings_batch(
    ["set-alarm", "check-weather"],
    [np.array([0.5, 0.2, 0.7, 0.1]), np.array([0.2, 0.6, 0.2, 0.8])],
    metadata=[{"intent": "alarm.set"}, {"intent": "weather.query"}],
)
print(f"\nUtterances after batch add: {db.count_embeddings_in_collection('utterances')}")
