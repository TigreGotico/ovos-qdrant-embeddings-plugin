# Configuration

`QdrantEmbeddingsDB` is configured via a dictionary passed to its constructor (or via
the OPM plugin config system when loaded by OVOS).

## All configuration keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vector_size` | `int` | **required** | Dimension of every vector stored in this DB. Must match the output dimension of your embedding model. All collections share this size. |
| `distance_metric` | `str` | `"cosine"` | Similarity function used for nearest-neighbour search. One of `"cosine"`, `"euclidean"`, `"dot"`. |
| `default_collection_name` | `str` | `"embeddings"` | Name of the collection created automatically on startup. Used whenever `collection_name` is `None`. |
| `host` | `str` | none | Hostname of a remote Qdrant server. Setting this key activates **remote** client mode. |
| `port` | `int` | `6333` | HTTP REST port for the remote client. |
| `grpc_port` | `int` | `6334` | gRPC port for the remote client (used for high-throughput batch operations). |
| `api_key` | `str` | none | Authentication key for Qdrant Cloud or a secured self-hosted instance. |
| `path` | `str` | none | Filesystem directory for local persistent storage. Setting this key (without `host`) activates **local** client mode. |

## Client modes

The constructor inspects the config keys to decide which Qdrant client to create:

### In-memory (development / CI)

Neither `host` nor `path` is present:

```python
QdrantEmbeddingsDB(config={"vector_size": 384})
```

Data is lost when the object is garbage-collected. Ideal for tests and quick prototyping.

### Local persistent

`path` is set, `host` is absent:

```python
QdrantEmbeddingsDB(config={
    "path": "/var/lib/ovos/qdrant",
    "vector_size": 384,
})
```

Qdrant stores its WAL and segments under `path`. Survives restarts. No network required.

### Remote

`host` is set:

```python
QdrantEmbeddingsDB(config={
    "host": "qdrant.example.com",
    "port": 6333,
    "grpc_port": 6334,
    "api_key": "optional-secret",
    "vector_size": 384,
})
```

Connects over HTTP (REST) to the specified Qdrant server. Use `api_key` for Qdrant Cloud
or any instance with authentication enabled.

## Distance metrics

| Value | Qdrant enum | Best for |
|-------|-------------|----------|
| `"cosine"` | `Distance.COSINE` | Sentence or word embeddings. Direction matters, magnitude does not. |
| `"euclidean"` | `Distance.EUCLID` | Dense float vectors where absolute distance matters. |
| `"dot"` | `Distance.DOT` | Pre-normalized vectors. Equivalent to cosine, but faster. |

## Cosine normalization note

When `distance_metric` is `"cosine"`, Qdrant **normalizes every vector to unit length on
upsert**. Vectors retrieved through `get_embeddings` are unit-length, not the original
floats. Nearest-neighbour query results are unaffected, because direction is preserved.
Do not compare retrieved vectors to originals with `np.allclose`. Compare directions
instead:

```python
norm_v = v / np.linalg.norm(v)
norm_r = retrieved / np.linalg.norm(retrieved)
np.testing.assert_allclose(norm_v, norm_r, atol=1e-5)
```

---
[Home](../README.md) · [Usage →](usage.md)
