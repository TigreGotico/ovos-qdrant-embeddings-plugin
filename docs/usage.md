# Usage

## Collections

Qdrant organizes vectors into named **collections**. Each collection shares the same
`vector_size` and `distance_metric` configured at construction time.

A default collection (`default_collection_name`, default `"embeddings"`) is created
automatically on startup. Most operations accept an optional `collection_name`; when
omitted the default collection is used.

```python
from ovos_qdrant_embeddings import QdrantEmbeddingsDB

db = QdrantEmbeddingsDB(config={"vector_size": 4})

# Create additional collections
db.create_collection("skills")
db.create_collection("memories")

# List all collections
for col in db.list_collections():
    print(col.name)

# Delete a collection
db.delete_collection("memories")
```

## Adding vectors

### Single

```python
import numpy as np

db.add_embeddings(
    key="hello-world",
    embedding=np.array([0.1, 0.2, 0.3, 0.4]),
    metadata={"source": "utterance", "lang": "en-us"},
    collection_name="skills",   # omit to use default
)
```

The `key` is stored in the point payload as `original_key` and is used for all
retrieval and deletion operations.

### Batch

```python
keys = ["doc1", "doc2", "doc3"]
vecs = [np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0])]
metas = [{"topic": "A"}, {"topic": "B"}, {"topic": "C"}]

db.add_embeddings_batch(keys, vecs, metadata=metas)
```

## Retrieving vectors

### Single

```python
# Returns np.ndarray or None
emb = db.get_embeddings("hello-world", collection_name="skills")

# With metadata — returns (np.ndarray, dict) or (None, None)
emb, meta = db.get_embeddings("hello-world", collection_name="skills", return_metadata=True)
```

### Batch

```python
# Returns list of (key, embedding) or (key, embedding, metadata)
results = db.get_embeddings_batch(["doc1", "doc2"], return_metadata=True)
for key, emb, meta in results:
    print(key, meta)
```

## Querying — nearest-neighbour search

```python
query_vec = np.array([0.9, 0.1, 0.0, 0.0])

# Returns list of (key, score)
hits = db.query(query_vec, top_k=5)

# With metadata — returns list of (key, score, metadata)
hits = db.query(query_vec, top_k=5, return_metadata=True)
for key, score, meta in hits:
    print(f"{key}: {score:.4f}  {meta}")
```

Score semantics depend on `distance_metric`:
- **cosine**: higher is more similar (range −1 … 1, typically 0 … 1 for non-negative vectors).
- **euclidean**: lower is closer.
- **dot**: higher is more similar.

## Deleting vectors

### Single

```python
db.delete_embeddings("hello-world", collection_name="skills")
```

### Batch

```python
db.delete_embeddings_batch(["doc1", "doc2"])
```

## Counting

```python
n = db.count_embeddings_in_collection()           # default collection
n = db.count_embeddings_in_collection("skills")   # named collection
```

## Metadata

Arbitrary JSON-serializable metadata can be attached to any vector. The internal key
`original_key` is reserved — it is injected automatically and stripped from results
returned to the caller.

All metadata fields are stored in the Qdrant point payload and are returned verbatim
(minus `original_key`).
