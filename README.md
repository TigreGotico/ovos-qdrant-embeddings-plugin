[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ovos-qdrant-embeddings-plugin)

# QdrantEmbeddingsDB Plugin

## Overview

The `QdrantEmbeddingsDB` plugin integrates with the [qdrant](https://www.tryQdrant.com/) database to provide a robust solution for managing and querying embeddings. This plugin extends the abstract `EmbeddingsDB` class, allowing you to store, retrieve, and query embeddings efficiently using qdrant’s capabilities.

## Features

- **Add Embeddings**: Store embeddings with associated keys.
- **Retrieve Embeddings**: Fetch embeddings by their keys.
- **Delete Embeddings**: Remove embeddings by their keys.
- **Query Embeddings**: Find the closest embeddings to a given query, with support for cosine distance.

## Example

Here is a full example demonstrating the basic usage of `QdrantEmbeddingsDB`.

```python
# Define a storage path for testing local persistent client
TEST_DB_PATH = "qdrant_test_storage"

# Clean up previous test data if it exists
if os.path.exists(TEST_DB_PATH):
    print(f"Cleaned up existing test data at {TEST_DB_PATH}")
    shutil.rmtree(TEST_DB_PATH)

# --- Test with Local Persistent Client ---
print("\n--- Initializing QdrantEmbeddingsDB with Local Persistent Client ---")
# Note: vector_size is crucial for Qdrant
db_local = QdrantEmbeddingsDB(config=dict(path=TEST_DB_PATH,
                                          default_collection_name="my_local_embeddings",
                                          vector_size=3,  # Example vector size
                                          distance_metric="cosine"))
print(f"Default collection name: {db_local.default_collection_name}")

# Test collection management
print("\n--- Testing Collection Management (Local) ---")
new_collection_name_local = "my_new_local_collection"
db_local.create_collection(new_collection_name_local, metadata={"purpose": "local_testing"})
print(f"Created new local collection: {new_collection_name_local}")

collections_local = db_local.list_collections()
print("Available local collections:")
for col in collections_local:
    print(f"  - {col.name}")

# Add embeddings to the default local collection
print("\n--- Adding Embeddings to Default Local Collection ---")
embedding1 = np.array([0.1, 0.2, 0.3])
embedding2 = np.array([0.4, 0.5, 0.6])
embedding3 = np.array([0.7, 0.8, 0.9])

db_local.add_embeddings("user1", embedding1, metadata={"name": "Bob", "age": 30})
db_local.add_embeddings("user2", embedding2, metadata={"name": "Joe", "city": "New York"})
print("Added user1 and user2 embeddings to default local collection.")

# Add embeddings to the new local collection
print("\n--- Adding Embeddings to New Local Collection ---")
db_local.add_embeddings("itemA", embedding1 * 0.5, metadata={"type": "product"},
                        collection_name=new_collection_name_local)
db_local.add_embeddings("itemB", embedding2 * 0.5, metadata={"type": "service"},
                        collection_name=new_collection_name_local)
print("Added itemA and itemB embeddings to new_local_collection.")

# Test count_embeddings_in_collection
print("\n--- Testing Embedding Count (Local) ---")
print(f"Embeddings in default local collection: {db_local.count_embeddings_in_collection()}")
print(
    f"Embeddings in '{new_collection_name_local}' local collection: {db_local.count_embeddings_in_collection(new_collection_name_local)}")

# Retrieve and print embeddings from default local collection
print("\n--- Retrieving Embeddings from Default Local Collection ---")
retrieved_emb1 = db_local.get_embeddings("user1")
print(f"Retrieved embedding for user1 (no metadata): {retrieved_emb1}")
retrieved_emb1_meta, retrieved_meta1 = db_local.get_embeddings("user1", return_metadata=True)
print(f"Retrieved embedding and metadata for user1: {retrieved_emb1_meta}, {retrieved_meta1}")

# Test batch add and get
print("\n--- Testing Batch Operations (Local) ---")
batch_keys = ["batch_user3", "batch_user4"]
batch_embeddings = [np.array([0.9, 0.8, 0.7]), np.array([0.6, 0.5, 0.4])]
batch_metadata = [{"source": "batch_upload"}, {"source": "batch_upload", "tag": "test"}]
db_local.add_embeddings_batch(batch_keys, batch_embeddings, batch_metadata)
print("Added batch_user3 and batch_user4 via batch operation to local collection.")

retrieved_batch = db_local.get_embeddings_batch(batch_keys, return_metadata=True)
print("Retrieved batch embeddings (with metadata) from local collection:")
for key, emb, meta in retrieved_batch:
    print(f"  Key: {key}, Embedding: {emb}, Metadata: {meta}")

# Query embeddings in default local collection
print("\n--- Querying Embeddings in Default Local Collection ---")
query_embedding = np.array([0.2, 0.3, 0.4])
results = db_local.query(query_embedding, top_k=2)
print(f"Query results (no metadata): {results}")
results_with_meta = db_local.query(query_embedding, top_k=2, return_metadata=True)
print(f"Query results (with metadata): {results_with_meta}")

# Test batch delete
print("\n--- Testing Batch Delete (Local) ---")
db_local.delete_embeddings_batch(["batch_user3", "user1"])
print("Deleted batch_user3 and user1 via batch delete from local collection.")
print(f"Embeddings in default local collection after batch delete: {db_local.count_embeddings_in_collection()}")
print(f"Retrieved embedding for user1 after delete: {db_local.get_embeddings('user1')}")  # Should be None

# Delete an embedding from the new local collection
print("\n--- Deleting Embeddings from New Local Collection ---")
db_local.delete_embeddings("itemA", collection_name=new_collection_name_local)
print("Deleted itemA from new_local_collection.")
print(
    f"Embeddings in '{new_collection_name_local}' local collection after delete: {db_local.count_embeddings_in_collection(new_collection_name_local)}")

# Delete the new local collection
print("\n--- Deleting New Local Collection ---")
db_local.delete_collection(new_collection_name_local)
collections_after_delete_local = db_local.list_collections()
print("Available collections after deleting new_local_collection:")
for col in collections_after_delete_local:
    print(f"  - {col.name}")
if not any(col.name == new_collection_name_local for col in collections_after_delete_local):
    print(f"Collection '{new_collection_name_local}' successfully deleted.")
else:
    print(f"Collection '{new_collection_name_local}' still exists (unexpected).")

# Clean up test data
if os.path.exists(TEST_DB_PATH):
    shutil.rmtree(TEST_DB_PATH)
    print(f"\nCleaned up test data at {TEST_DB_PATH}")

# --- Test with In-Memory Client ---
print("\n--- Initializing QdrantEmbeddingsDB with In-Memory Client ---")
db_memory = QdrantEmbeddingsDB(config=dict(default_collection_name="my_memory_embeddings",
                                           vector_size=3,  # Example vector size
                                           distance_metric="euclidean"))
print(f"Default collection name: {db_memory.default_collection_name}")

# Add embeddings to the default in-memory collection
print("\n--- Adding Embeddings to Default In-Memory Collection ---")
db_memory.add_embeddings("mem_user1", np.array([1.0, 1.0, 1.0]), metadata={"source": "memory_test"})
db_memory.add_embeddings("mem_user2", np.array([2.0, 2.0, 2.0]))
print("Added mem_user1 and mem_user2 embeddings to default in-memory collection.")

print(f"Embeddings in default in-memory collection: {db_memory.count_embeddings_in_collection()}")

# Query in-memory collection
print("\n--- Querying Embeddings in Default In-Memory Collection ---")
query_memory_embedding = np.array([1.1, 1.1, 1.1])
results_memory = db_memory.query(query_memory_embedding, top_k=1, return_metadata=True)
print(f"Query results (with metadata) in-memory: {results_memory}")

# --- Example of Remote Client (uncomment and configure if you have a Qdrant server) ---
# print("\n--- Initializing QdrantEmbeddingsDB with Remote Client (requires running Qdrant server) ---")
# try:
#     db_remote = QdrantEmbeddingsDB(config=dict(host="localhost", # Replace with your Qdrant host
#                                                 port=6333,
#                                                 default_collection_name="my_remote_embeddings",
#                                                 vector_size=3, # Must match your remote collection's vector size
#                                                 distance_metric="cosine"))
#     print(f"Default remote collection name: {db_remote.default_collection_name}")
#     # Test remote operations (similar to local tests)
#     db_remote.add_embeddings("remote_user1", np.array([0.1, 0.2, 0.3]), metadata={"source": "remote_test"})
#     print(f"Embeddings in default remote collection: {db_remote.count_embeddings_in_collection()}")
#     remote_results = db_remote.query(np.array([0.15, 0.25, 0.35]), top_k=1)
#     print(f"Remote query results: {remote_results}")
#     db_remote.delete_embeddings("remote_user1")
#     print(f"Embeddings in default remote collection after delete: {db_remote.count_embeddings_in_collection()}")
# except Exception as e:
#     print(f"Could not connect to remote Qdrant server. Please ensure it's running. Error: {e}")

```

> Ensure that the path provided to the `QdrantEmbeddingsDB` constructor is accessible and writable.

