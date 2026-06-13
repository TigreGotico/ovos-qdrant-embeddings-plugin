"""
Network-free tests for QdrantEmbeddingsDB using in-memory Qdrant client.
"""
import numpy as np
import pytest

from ovos_qdrant_embeddings import QdrantEmbeddingsDB

VECTOR_SIZE = 4


@pytest.fixture
def db():
    """Fresh in-memory DB for each test."""
    return QdrantEmbeddingsDB(config={"vector_size": VECTOR_SIZE})


def vec(*values):
    """Helper: make a float32 numpy array of length VECTOR_SIZE."""
    return np.array(values, dtype=np.float32)


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_default_collection(db):
    names = [c.name for c in db.list_collections()]
    assert db.default_collection_name in names


def test_init_requires_vector_size():
    with pytest.raises(ValueError, match="vector_size"):
        QdrantEmbeddingsDB(config={})


# ── add / get ─────────────────────────────────────────────────────────────────

def test_add_and_get_embeddings(db):
    v = vec(0.1, 0.2, 0.3, 0.4)
    db.add_embeddings("k1", v)
    result = db.get_embeddings("k1")
    assert result is not None
    assert len(result) == VECTOR_SIZE
    # Qdrant normalizes cosine vectors; verify same direction via dot product ~ 1
    norm_v = v / np.linalg.norm(v)
    norm_r = result / np.linalg.norm(result)
    np.testing.assert_allclose(norm_v, norm_r, atol=1e-5)


def test_get_nonexistent_key_returns_none(db):
    assert db.get_embeddings("does_not_exist") is None


def test_get_embeddings_return_metadata(db):
    v = vec(0.1, 0.2, 0.3, 0.4)
    db.add_embeddings("k2", v, metadata={"label": "test"})
    emb, meta = db.get_embeddings("k2", return_metadata=True)
    assert emb is not None
    assert len(emb) == VECTOR_SIZE
    # Qdrant normalizes cosine vectors; verify same direction
    norm_v = v / np.linalg.norm(v)
    norm_e = emb / np.linalg.norm(emb)
    np.testing.assert_allclose(norm_v, norm_e, atol=1e-5)
    assert meta.get("label") == "test"
    # original_key should not leak into returned metadata
    assert "original_key" not in meta


def test_get_embeddings_no_metadata_tuple(db):
    """return_metadata=False (default) returns the array, not a tuple."""
    v = vec(0.5, 0.6, 0.7, 0.8)
    db.add_embeddings("k3", v)
    result = db.get_embeddings("k3", return_metadata=False)
    assert isinstance(result, np.ndarray)


# ── batch add / get ───────────────────────────────────────────────────────────

def test_add_embeddings_batch_and_get_batch(db):
    keys = ["b1", "b2", "b3"]
    vecs = [vec(1, 0, 0, 0), vec(0, 1, 0, 0), vec(0, 0, 1, 0)]
    db.add_embeddings_batch(keys, vecs)

    results = db.get_embeddings_batch(keys)
    found_keys = {r[0] for r in results}
    assert found_keys == set(keys)
    # Each result is (key, embedding)
    for key, emb in results:
        assert isinstance(emb, np.ndarray)
        assert len(emb) == VECTOR_SIZE


def test_add_embeddings_batch_with_metadata(db):
    keys = ["bm1", "bm2"]
    vecs = [vec(1, 1, 0, 0), vec(0, 0, 1, 1)]
    metas = [{"tag": "alpha"}, {"tag": "beta"}]
    db.add_embeddings_batch(keys, vecs, metadata=metas)

    results = db.get_embeddings_batch(keys, return_metadata=True)
    assert len(results) == 2
    tag_map = {r[0]: r[2].get("tag") for r in results}
    assert tag_map.get("bm1") == "alpha"
    assert tag_map.get("bm2") == "beta"


def test_get_embeddings_batch_without_metadata(db):
    keys = ["nb1", "nb2"]
    vecs = [vec(0.1, 0.1, 0.1, 0.1), vec(0.9, 0.9, 0.9, 0.9)]
    db.add_embeddings_batch(keys, vecs)

    results = db.get_embeddings_batch(keys, return_metadata=False)
    assert len(results) == 2
    for item in results:
        assert len(item) == 2  # (key, embedding)


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_top_k_returns_ids_and_distances(db):
    db.add_embeddings("q1", vec(1, 0, 0, 0))
    db.add_embeddings("q2", vec(0, 1, 0, 0))
    db.add_embeddings("q3", vec(0, 0, 1, 0))

    results = db.query(vec(1, 0, 0, 0), top_k=2)
    assert len(results) == 2
    for item in results:
        assert len(item) == 2
        key, dist = item
        assert isinstance(key, str)
        assert isinstance(dist, float)
    # Closest match should be q1
    assert results[0][0] == "q1"


def test_query_with_metadata(db):
    db.add_embeddings("qm1", vec(1, 0, 0, 0), metadata={"info": "x"})
    db.add_embeddings("qm2", vec(0, 1, 0, 0), metadata={"info": "y"})

    results = db.query(vec(1, 0, 0, 0), top_k=1, return_metadata=True)
    assert len(results) == 1
    key, dist, meta = results[0]
    assert key == "qm1"
    assert isinstance(meta, dict)
    assert "original_key" not in meta


# ── collection lifecycle ──────────────────────────────────────────────────────

def test_create_and_list_collections(db):
    db.create_collection("col_a")
    db.create_collection("col_b")
    names = [c.name for c in db.list_collections()]
    assert "col_a" in names
    assert "col_b" in names


def test_delete_collection(db):
    db.create_collection("to_delete")
    db.delete_collection("to_delete")
    names = [c.name for c in db.list_collections()]
    assert "to_delete" not in names


def test_count_embeddings(db):
    assert db.count_embeddings_in_collection() == 0
    db.add_embeddings("c1", vec(0.1, 0.2, 0.3, 0.4))
    db.add_embeddings("c2", vec(0.5, 0.6, 0.7, 0.8))
    assert db.count_embeddings_in_collection() == 2


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_embeddings(db):
    db.add_embeddings("del1", vec(0.1, 0.2, 0.3, 0.4))
    db.delete_embeddings("del1")
    assert db.get_embeddings("del1") is None


def test_delete_embeddings_batch(db):
    keys = ["db1", "db2", "db3"]
    vecs = [vec(1, 0, 0, 0), vec(0, 1, 0, 0), vec(0, 0, 1, 0)]
    db.add_embeddings_batch(keys, vecs)

    db.delete_embeddings_batch(["db1", "db2"])
    assert db.get_embeddings("db1") is None
    assert db.get_embeddings("db2") is None
    # db3 should still exist
    assert db.get_embeddings("db3") is not None
