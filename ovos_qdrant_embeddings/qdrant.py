import uuid
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
from ovos_plugin_manager.templates.embeddings import EmbeddingsDB, EmbeddingsArray, EmbeddingsTuple, \
    RetrievedEmbeddingResult
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    FilterSelector,
    CollectionInfo,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny
)


class QdrantEmbeddingsDB(EmbeddingsDB):
    """An implementation of EmbeddingsDB using Qdrant for managing embeddings."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the QdrantEmbeddingsDB.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
                Expected keys:
                - "host": (str) Host for remote Qdrant client.
                - "port": (int) Port for remote Qdrant client (default: 6333).
                - "grpc_port": (int) gRPC port for remote Qdrant client (default: 6334).
                - "api_key": (str) API key for remote Qdrant client.
                - "path": (str) Path for local persistent Qdrant client.
                - "default_collection_name": (str) Name of the default collection (default: "embeddings").
                - "vector_size": (int) Required. The dimension of the embedding vectors.
                - "distance_metric": (str) The distance metric to use (default: "cosine").
                                     Supported: "cosine", "euclidean", "dot".
        """
        super().__init__(config)

        # Determine the default collection name from config, or use "embeddings"
        self.default_collection_name = self.config.get("default_collection_name", "embeddings")

        # Get vector size and distance metric from config
        self.vector_size = self.config.get("vector_size")
        if self.vector_size is None:
            raise ValueError("QdrantEmbeddingsDB requires 'vector_size' in its configuration.")

        distance_metric_str = self.config.get("distance_metric", "cosine").lower()
        self.distance_metric = self._map_distance_metric(distance_metric_str)

        # Initialize Qdrant client based on host (for HTTP client) or path (for Persistent client)
        if "host" in self.config:
            # Remote client
            host = self.config["host"]
            port = self.config.get("port", 6333)
            grpc_port = self.config.get("grpc_port", 6334)
            api_key = self.config.get("api_key")
            self.client = QdrantClient(host=host, port=port, grpc_port=grpc_port, api_key=api_key)
            print(f"Initialized remote Qdrant client at {host}:{port}")
        elif "path" in self.config:
            # Local persistent client
            path = self.config["path"]
            self.client = QdrantClient(path=path)
            print(f"Initialized local persistent Qdrant client at {path}")
        else:
            # In-memory client (for testing/development)
            self.client = QdrantClient(":memory:")
            print("Initialized in-memory Qdrant client")

        # Ensure the default collection exists upon initialization
        self._get_or_create_collection(self.default_collection_name)

    def _map_distance_metric(self, metric_str: str) -> Distance:
        """Maps a string metric name to Qdrant's Distance enum."""
        if metric_str == "cosine":
            return Distance.COSINE
        elif metric_str == "euclidean":
            return Distance.EUCLID
        elif metric_str == "dot":
            return Distance.DOT
        else:
            raise ValueError(f"Unsupported distance metric: {metric_str}. Choose from 'cosine', 'euclidean', 'dot'.")

    def _get_collection_instance(self, collection_name: Optional[str]) -> CollectionInfo:
        """
        Helper method to get the Qdrant collection info.
        If collection_name is None, returns the default collection.
        Raises ValueError if the collection does not exist.
        """
        name = collection_name or self.default_collection_name
        try:
            # Check if collection exists
            if not self.client.collection_exists(collection_name=name):
                raise ValueError(f"Collection '{name}' not found.")
            return self.client.get_collection(collection_name=name)
        except Exception as e:
            raise ValueError(f"Collection '{name}' not found or could not be accessed: {e}")

    def _get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Internal helper to get or create a collection.
        Qdrant's create_collection handles the logic of creating if not exists.
        """
        if not self.client.collection_exists(collection_name=name):
            print(
                f"Creating collection '{name}' with vector size {self.vector_size} and distance {self.distance_metric.value}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance_metric),
                # Qdrant metadata is per-point, not per-collection, but we can store it here
                # if there's a need to retrieve collection-level metadata later.
                # For now, we just ensure the collection exists.
            )
            # You might want to store collection-level metadata in a special point or separate system
            # if Qdrant doesn't directly support it at the collection level in a way you need.
        else:
            print(f"Collection '{name}' already exists.")
        return self.client.get_collection(collection_name=name)

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a new collection (vector store).
        Args:
            name (str): The name of the collection (vector store ID).
            metadata (Optional[Dict[str, Any]]): Optional metadata for the collection.
                                                 Note: Qdrant stores metadata per point, not per collection.
        Returns:
            Any: A handle or object representing the created collection (CollectionInfo).
        """
        return self._get_or_create_collection(name, metadata)

    def get_collection(self, name: str) -> Any:
        """
        Retrieve an existing collection by name.
        Args:
            name (str): The name of the collection.
        Returns:
            Any: A handle or object representing the retrieved collection (CollectionInfo).
        Raises:
            ValueError: If the collection is not found.
        """
        return self._get_collection_instance(name)

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection by name.
        Args:
            name (str): The name of the collection to delete.
        """
        try:
            self.client.delete_collection(collection_name=name)
            print(f"Collection '{name}' deleted.")
        except Exception as e:
            print(f"Error deleting collection '{name}': {e}")

    def list_collections(self) -> List[Any]:
        """
        List all available collections.
        Returns:
            List[Any]: A list of handles or objects representing the collections (CollectionInfo).
        """
        collections = self.client.get_collections().collections
        return collections

    def add_embeddings(self, key: str, embedding: EmbeddingsArray,
                       metadata: Optional[Dict[str, Any]] = None,
                       collection_name: Optional[str] = None) -> EmbeddingsArray:
        """
        Store 'embedding' under 'key' with associated metadata in the specified or default collection.
        Args:
            key (str): The unique key for the embedding. This will be stored as 'original_key' in payload.
            embedding (np.ndarray): The embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Optional metadata associated with the embedding.
            collection_name (Optional[str]): The name of the collection to add the embedding to.
                                             If None, a default collection should be used.
        Returns:
            np.ndarray: The stored embedding.
        """
        name = collection_name or self.default_collection_name
        # Ensure collection exists before adding points
        self._get_or_create_collection(name)

        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        # Qdrant expects payload (metadata) to be a dictionary
        payload = metadata if metadata is not None else {}
        # Store the original key in the payload
        payload["original_key"] = key

        # Generate a UUID for the Qdrant point ID
        point_id = str(uuid.uuid4())

        # Use PointStruct for upserting a single point
        point = PointStruct(id=point_id, vector=embedding_list, payload=payload)
        self.client.upsert(
            collection_name=name,
            points=[point],
            wait=True  # Wait for the operation to complete
        )
        return embedding

    def add_embeddings_batch(self, keys: List[str], embeddings: List[EmbeddingsArray],
                             metadata: Optional[List[Dict[str, Any]]] = None,
                             collection_name: Optional[str] = None) -> None:
        """
        Add or update multiple embeddings in a batch to a specific collection.
        Args:
            keys (List[str]): List of unique keys for the embeddings. These will be stored as 'original_key' in payload.
            embeddings (List[EmbeddingsArray]): List of embedding vectors to store.
            metadata (Optional[List[Dict[str, Any]]]): Optional list of metadata dictionaries.
            collection_name (Optional[str]): The name of the collection to add the embeddings to.
        """
        name = collection_name or self.default_collection_name
        # Ensure collection exists before adding points
        self._get_or_create_collection(name)

        points = []
        for i, key in enumerate(keys):
            embedding_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
            payload = metadata[i] if metadata and metadata[i] is not None else {}
            # Store the original key in the payload
            payload["original_key"] = key
            # Generate a UUID for the Qdrant point ID
            point_id = str(uuid.uuid4())
            points.append(PointStruct(id=point_id, vector=embedding_list, payload=payload))

        self.client.upsert(
            collection_name=name,
            points=points,
            wait=True
        )

    def get_embeddings(self, key: str, collection_name: Optional[str] = None,
                       return_metadata: bool = False) -> Union[Optional[EmbeddingsArray],
    Tuple[Optional[EmbeddingsArray], Optional[Dict[str, Any]]]]:
        """
        Retrieve embeddings stored under 'key' from the specified or default collection.

        Args:
            key (str): The unique key for the embedding (original_key in payload).
            collection_name (Optional[str]): The name of the collection to retrieve from.
            return_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.

        Returns:
            Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]] :
            If `return_metadata` is False, returns the retrieved embedding (np.ndarray) or None if not found.
            If `return_metadata` is True, returns a tuple (embedding, metadata_dict) or (None, None) if not found.
        """
        name = collection_name or self.default_collection_name
        try:
            # Create a filter to find the point by its original_key in the payload
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="original_key",
                        match=MatchValue(value=key)
                    )
                ]
            )

            # Use scroll to retrieve points based on filter
            # Limit to 1 as we expect only one result for a unique key
            retrieved_points, _ = self.client.scroll(
                collection_name=name,
                scroll_filter=search_filter,
                limit=1,
                with_vectors=True,
                with_payload=True
            )

            if retrieved_points:
                point = retrieved_points[0]
                embedding_array = np.array(point.vector)
                # Remove 'original_key' from metadata if returning metadata
                payload = point.payload or {}
                if "original_key" in payload:
                    del payload["original_key"]

                if return_metadata:
                    return embedding_array, payload
                return embedding_array
            return None if not return_metadata else (None, None)
        except Exception as e:
            print(f"Error retrieving embedding '{key}' from collection '{name}': {e}")
            return None if not return_metadata else (None, None)

    def get_embeddings_batch(self, keys: List[str], collection_name: Optional[str] = None,
                             return_metadata: bool = False) -> List[RetrievedEmbeddingResult]:
        """
        Retrieve multiple embeddings and their metadata from a specific collection.

        Args:
            keys (List[str]): List of keys for the embeddings to retrieve (original_key in payload).
            collection_name (Optional[str]): The name of the collection to retrieve from.
            return_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.

        Returns:
            List[RetrievedEmbeddingResult]: A list of tuples, where each tuple is
            (key, embedding) if `return_metadata` is False, or (key, embedding, metadata)
            if `return_metadata` is True.
        """
        name = collection_name or self.default_collection_name
        retrieved_embeddings = []
        try:
            # Create a filter to find points by their original_key in the payload
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="original_key",
                        match=MatchAny(any=keys)  # Match any of the provided keys
                    )
                ]
            )

            # Use scroll to retrieve points based on filter
            # No limit, retrieve all matching points
            retrieved_points, _ = self.client.scroll(
                collection_name=name,
                scroll_filter=search_filter,
                limit=len(keys),  # Limit to the number of keys requested
                with_vectors=True,
                with_payload=True
            )

            for point in retrieved_points:
                original_key = point.payload.get("original_key")
                if original_key:  # Ensure original_key exists
                    embedding = np.array(point.vector)
                    payload = point.payload or {}
                    if "original_key" in payload:
                        del payload["original_key"]  # Remove original_key from metadata returned to user

                    if return_metadata:
                        retrieved_embeddings.append((original_key, embedding, payload))
                    else:
                        retrieved_embeddings.append((original_key, embedding))
        except Exception as e:
            print(f"Error retrieving batch embeddings from collection '{name}': {e}")
            # Continue with partial results or empty list
        return retrieved_embeddings

    def delete_embeddings(self, key: str, collection_name: Optional[str] = None) -> None:
        """
        Delete embeddings stored under 'key' from the specified or default collection.
        Args:
            key (str): The unique key for the embedding (original_key in payload).
            collection_name (Optional[str]): The name of the collection to delete from.
        """
        name = collection_name or self.default_collection_name
        try:
            # Create a filter to find the point by its original_key in the payload
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="original_key",
                        match=MatchValue(value=key)
                    )
                ]
            )
            self.client.delete(
                collection_name=name,
                points_selector=FilterSelector(filter=delete_filter),
                wait=True
            )
            print(f"Deleted embedding with original_key '{key}' from collection '{name}'.")
        except Exception as e:
            print(f"Error deleting embedding with original_key '{key}' from collection '{name}': {e}")

    def delete_embeddings_batch(self, keys: List[str], collection_name: Optional[str] = None) -> None:
        """
        Delete multiple embeddings in a batch from a specific collection.
        Args:
            keys (List[str]): List of keys for the embeddings to delete (original_key in payload).
            collection_name (Optional[str]): The name of the collection to delete from.
        """
        name = collection_name or self.default_collection_name
        try:
            # Create a filter to find points by their original_key in the payload
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="original_key",
                        match=MatchAny(any=keys)  # Match any of the provided keys
                    )
                ]
            )
            self.client.delete(
                collection_name=name,
                points_selector=FilterSelector(filter=delete_filter),
                wait=True
            )
            print(f"Deleted batch embeddings with original_keys {keys} from collection '{name}'.")
        except Exception as e:
            print(f"Error deleting batch embeddings from collection '{name}': {e}")

    def query(self, embeddings: EmbeddingsArray, top_k: int = 5,
              return_metadata: bool = False, collection_name: Optional[str] = None) -> List[EmbeddingsTuple]:
        """
        Query the database for the closest embeddings to the given query embedding
        in the specified or default collection.
        Args:
            embeddings (np.ndarray): The embedding vector to query.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            return_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.
            collection_name (Optional[str]): The name of the collection to query.
        Returns:
            List[EmbeddingsTuple]: A list of tuples containing the keys, distances, and optionally metadata.
        """
        name = collection_name or self.default_collection_name
        embedding_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        search_result = self.client.query_points(
            name,  # collection_name as first positional argument
            embedding_list,  # query_vector as second positional argument
            limit=top_k,
            with_payload=True,  # Always request payload to get original_key
            with_vectors=False  # No need to return vectors themselves for query results
        )

        results_list = []
        for scored_point in search_result.points:
            if scored_point:  # Ensure we have a valid scored_point
                # Extract the original_key from the payload
                original_key = scored_point.payload.get("original_key")
                if original_key is None:
                    # If original_key is not found, use the Qdrant internal ID as a fallback
                    original_key = str(scored_point.id)

                distance = scored_point.score  # Qdrant returns score, which is typically distance or similarity

                # Prepare metadata to return, excluding the internal original_key
                metadata = scored_point.payload or {}
                if "original_key" in metadata:
                    del metadata["original_key"]

                if return_metadata:
                    results_list.append((original_key, float(distance), metadata))
                else:
                    results_list.append((original_key, float(distance)))
        return results_list

    def count_embeddings_in_collection(self, collection_name: Optional[str] = None) -> int:
        """
        Count the number of embeddings (points) in a specific collection.
        Args:
            collection_name (Optional[str]): The name of the collection.
        Returns:
            int: The number of embeddings in the collection.
        """
        name = collection_name or self.default_collection_name
        try:
            count_result = self.client.count(
                collection_name=name,
                exact=True  # Get exact count
            )
            return count_result.count
        except Exception as e:
            print(f"Error counting embeddings in collection '{name}': {e}")
            return 0


if __name__ == "__main__":
    import os
    import shutil

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
