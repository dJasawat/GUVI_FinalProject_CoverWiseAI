from qdrant_client import QdrantClient, models
import os
#from policy_analyzer import policy_chunks
import numpy as np
import uuid
from dotenv import load_dotenv

load_dotenv()

POLICY_COLLECTION_NAME = "insurance_policy_chunks"
INSURANCE_QA_PAIRS_COLLECTION_NAME = "insurancePolicy_QA_Pairs"
# ger environment variables for Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

#create qdrant client

def get_qdrant_client(qdrant_url, qdrant_api_key):
    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

#Method Check If collection exists in Qdrant if not create Collection
def collection_exists( qdrant_client: QdrantClient, collection_name: str, vectorsDimensions: int) -> bool:
    try:
        # Check if collection exists, create if not
        if not qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                                        size=vectorsDimensions,
                                        distance=models.Distance.COSINE),
            )
            #print(f"Collection '{collection_name}' created.")
        #else:
            #print(f"Collection '{collection_name}' already exists.")

    except Exception as e:
        print(f"Error creating or checking collection '{collection_name}': {e}")
        return False

def save_PolicyDocs_to_qdrant(qdrant_url: str, qdrant_api_key: str,document_id: str, vectors: np.ndarray, all_chunks: list):
    collection_name = POLICY_COLLECTION_NAME
    qdrant_client = get_qdrant_client(qdrant_url, qdrant_api_key)
    collection_exists(qdrant_client,collection_name, vectors.shape[1])
    
    # Prepare points for upsert
   # document_id = str(uuid.uuid4())
    points = []
    for i, vector in enumerate(vectors):
        chunk = all_chunks[i]
        payload = {
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "policy_name": chunk.policy_name,
            "page_number": chunk.page_number,
            "page_numbers": chunk.page_numbers,
            "clause_heading": chunk.clause_heading,
            "section_title": chunk.section_title,
            "text": chunk.text
         }
        point_id = str(
            uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{chunk.policy_name}_{chunk.chunk_id}")
          )
        points.append(
            models.PointStruct(
                id=point_id,  # Unique ID for each chunk
                vector=vector.tolist(), # Qdrant expects list, not np.ndarray
                payload=payload
            )
        )
    try:
        # Upsert points
        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        print(f"Upsert operation info: {operation_info}")
        print(f"Successfully saved {len(points)} embeddings to Qdrant collection '{collection_name}'.")

    except Exception as e:
        print(f"Error upserting vectors to collection '{collection_name}': {e}")


#search policy chunks in Qdrant based on query vector - QA pairs only 
def search_policy_chunks(
    #collection_Name: str,
    query_vector: list[float],
    limit: int = 8,
) -> list[dict]:
    """Find the policy chunks most relevant to an embedded question."""
    qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    print(QDRANT_URL, QDRANT_API_KEY)
    if not qdrant_client.collection_exists(INSURANCE_QA_PAIRS_COLLECTION_NAME):
        return []
    
    print(QDRANT_URL, QDRANT_API_KEY)
    results = qdrant_client.query_points(
        collection_name=INSURANCE_QA_PAIRS_COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points
    return [
        {**(point.payload or {}), "score": point.score}
        for point in results
    ]

