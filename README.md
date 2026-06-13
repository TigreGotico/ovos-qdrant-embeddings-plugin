[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenVoiceOS/ovos-qdrant-embeddings-plugin)

# ovos-qdrant-embeddings-plugin

Qdrant-backed `EmbeddingsDB` vector store plugin for [OpenVoiceOS](https://openvoiceos.org).

## Install

```bash
pip install ovos-qdrant-embeddings-plugin
```

## What is EmbeddingsDB?

`EmbeddingsDB` is the abstract vector-store interface defined by
[ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos_plugin_manager) (`opm.embeddings`).
It provides a uniform API for storing, retrieving, and nearest-neighbour querying of embedding
vectors regardless of the backend.

OVOS components that need semantic search (e.g. memory, skills, pipelines) discover the active
backend via the OPM entry-point group `opm.embeddings`. This plugin registers itself as:

```
opm.embeddings = ovos-qdrant-embeddings-plugin
```

It pairs naturally with embedding producers such as
[ovos-gguf-plugin](https://github.com/OpenVoiceOS/ovos-gguf-plugin) that generate the vectors
you store here.

## Quickstart — in-memory DB

```python
import numpy as np
from ovos_qdrant_embeddings import QdrantEmbeddingsDB

# In-memory: no host, no path — perfect for development and CI
db = QdrantEmbeddingsDB(config={"vector_size": 4})

# Store vectors
db.add_embeddings("apple",  np.array([1.0, 0.0, 0.0, 0.0]))
db.add_embeddings("banana", np.array([0.0, 1.0, 0.0, 0.0]))
db.add_embeddings("cherry", np.array([0.0, 0.0, 1.0, 0.0]))

# Nearest-neighbour query — returns [(key, score), ...]
results = db.query(np.array([0.9, 0.1, 0.0, 0.0]), top_k=2)
print(results)  # [('apple', 0.999...), ('banana', 0.099...)]
```

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `vector_size` | **required** | Dimension of the embedding vectors. Must match your embedding model. |
| `distance_metric` | `"cosine"` | Similarity function: `"cosine"`, `"euclidean"`, or `"dot"`. |
| `default_collection_name` | `"embeddings"` | Collection created on startup and used when no collection is specified. |
| `host` | — | Remote Qdrant host (activates HTTP client mode). |
| `port` | `6333` | HTTP port for remote client. |
| `grpc_port` | `6334` | gRPC port for remote client. |
| `api_key` | — | API key for Qdrant Cloud or authenticated remote instances. |
| `path` | — | Filesystem path for local persistent storage (activates file-backed mode). |

### Three client modes

**In-memory** (development / CI) — neither `host` nor `path` set:
```python
config = {"vector_size": 384}
```

**Local persistent** — data survives restarts:
```python
config = {"path": "/var/lib/ovos/qdrant", "vector_size": 384}
```

**Remote** — connects to a running Qdrant server or Qdrant Cloud:
```python
config = {
    "host": "my-qdrant.example.com",
    "port": 6333,
    "api_key": "my-secret-key",
    "vector_size": 384,
}
```

## When to choose Qdrant over ChromaDB

- You need to run the vector store as a **separate network service** (microservice / homelab).
- Your collection grows to **millions of vectors** — Qdrant's HNSW index scales well.
- You want **Qdrant Cloud** managed hosting.
- You need **gRPC** for high-throughput batch ingestion.

For a single-device OVOS installation with moderate data, the ChromaDB plugin may be simpler.
Both expose the same `EmbeddingsDB` interface, so switching is a config change.

## Further reading

- [`docs/configuration.md`](docs/configuration.md) — all config keys, client modes, cosine normalization note
- [`docs/usage.md`](docs/usage.md) — collections, CRUD, batch ops, metadata, query
- [`examples/quickstart.py`](examples/quickstart.py) — in-memory add + query
- [`examples/collections.py`](examples/collections.py) — multi-collection workflow
- [`examples/remote_server.py`](examples/remote_server.py) — remote Qdrant setup

## Testing

Tests use an in-memory Qdrant client — no server required.

```bash
pip install ovos-qdrant-embeddings-plugin[test]
pytest test/ -v
```

## Credits

Originally developed by [TigreGótico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org),
sponsored by VisioLab. Modernized under the [NGI0 Commons Fund](https://nlnet.nl/commonsfund) / [NLnet](https://nlnet.nl).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
