"""
End-to-end integration tests using in-memory Qdrant — no mocks, no server required.

Vectors are deterministic: generated from string hashes via seeded numpy,
so the test is reproducible in CI without any model downloads.
"""
import hashlib

import numpy as np
import pytest

from ovos_qdrant_embeddings import QdrantEmbeddingsDB

VECTOR_SIZE = 8


def _str_to_vec(text: str, size: int = VECTOR_SIZE) -> np.ndarray:
    """Deterministic unit-length embedding derived from string content."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def db():
    return QdrantEmbeddingsDB(config={"vector_size": VECTOR_SIZE})


# ── basic e2e flow ─────────────────────────────────────────────────────────────

def test_e2e_add_query_nearest(db):
    """Store named vectors and verify query returns the nearest by direction."""
    sentences = ["hello world", "good morning", "the weather is nice", "play some music"]
    for s in sentences:
        db.add_embeddings(s, _str_to_vec(s), metadata={"text": s})

    # Query with the exact vector for "hello world" — it must be the top hit
    query_vec = _str_to_vec("hello world")
    results = db.query(query_vec, top_k=2, return_metadata=True)

    assert len(results) == 2
    top_key, top_score, top_meta = results[0]
    assert top_key == "hello world"
    assert top_score > 0.99, f"Expected near-perfect cosine match, got {top_score}"
    assert top_meta.get("text") == "hello world"
    assert "original_key" not in top_meta


def test_e2e_batch_add_and_query(db):
    """Batch-insert vectors and assert all are retrievable and queryable."""
    items = {
        "cat": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dog": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "car": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "bus": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }
    keys = list(items.keys())
    vecs = [np.array(v, dtype=np.float32) for v in items.values()]
    metas = [{"word": k} for k in keys]

    db.add_embeddings_batch(keys, vecs, metadata=metas)

    assert db.count_embeddings_in_collection() == len(keys)

    # Every key must be individually retrievable
    for k in keys:
        emb, meta = db.get_embeddings(k, return_metadata=True)
        assert emb is not None, f"Missing embedding for '{k}'"
        assert meta.get("word") == k

    # Query for "cat" direction — should return "cat" first
    results = db.query(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert results[0][0] == "cat"


def test_e2e_second_collection(db):
    """Vectors in different collections are isolated."""
    db.create_collection("alt")

    # Default collection: axis-aligned unit vectors
    db.add_embeddings("x", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    db.add_embeddings("y", np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))

    # Alt collection: different vectors
    db.add_embeddings("p", np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32), collection_name="alt")
    db.add_embeddings("q", np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32), collection_name="alt")

    assert db.count_embeddings_in_collection() == 2
    assert db.count_embeddings_in_collection("alt") == 2

    # Query in default — should NOT see alt vectors
    query = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    default_results = db.query(query, top_k=3)
    default_keys = {r[0] for r in default_results}
    assert "p" not in default_keys
    assert "q" not in default_keys

    # Query in alt — should find "p"
    alt_results = db.query(query, top_k=1, collection_name="alt")
    assert alt_results[0][0] == "p"


def test_e2e_metadata_roundtrip(db):
    """Metadata is stored and returned without leaking internal keys."""
    meta_in = {"lang": "en-us", "skill": "weather", "score": 0.95, "tags": ["demo"]}
    db.add_embeddings("meta-test", _str_to_vec("meta-test"), metadata=meta_in)

    emb, meta_out = db.get_embeddings("meta-test", return_metadata=True)
    assert emb is not None
    assert meta_out["lang"] == "en-us"
    assert meta_out["skill"] == "weather"
    assert meta_out["score"] == pytest.approx(0.95)
    assert "original_key" not in meta_out


def test_e2e_delete_and_reinsert(db):
    """Delete a vector and reinsert under the same key — latest version wins."""
    v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    db.add_embeddings("slot", v1, metadata={"version": 1})
    db.delete_embeddings("slot")
    assert db.get_embeddings("slot") is None

    db.add_embeddings("slot", v2, metadata={"version": 2})
    emb, meta = db.get_embeddings("slot", return_metadata=True)
    assert emb is not None
    assert meta.get("version") == 2
    # Direction should match v2 (cosine normalization preserves direction)
    norm_v2 = v2 / np.linalg.norm(v2)
    norm_emb = emb / np.linalg.norm(emb)
    np.testing.assert_allclose(norm_v2, norm_emb, atol=1e-5)
