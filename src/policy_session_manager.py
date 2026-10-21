from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer
#from paddleocr import PaddleOCRVL
#from src.policySchema import PolicySchema
import src.policy_analyzer as PolicyAnalyzer


class PolicySessionManager:
    """
    Manages uploaded health insurance policies entirely in memory.

    Policy data lives only in the current Streamlit session.
    """

    def __init__(
        self,
        embedding_model,
        chunk_size: int = 1200,
        chunk_overlap: int = 160,
    ):
      #  self.ocr = ocr
       # self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_model = embedding_model

        # policy_name -> policy data
        self.policies: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------

    @property
    def embedding_model(self):
        """
        Load SentenceTransformer only when required.
        """

        if self._embedding_model is None:

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )

        return self._embedding_model

    # ---------------------------------------------------------
    # Add policy
    # ---------------------------------------------------------

    def add_policy(
        self,
        uploaded_file,
    ) -> dict[str, Any]:
        """
        Process an uploaded Streamlit PDF and keep
        the resulting policy entirely in memory.
        """

        file_name = uploaded_file.name

        # Avoid processing the same file twice
        if file_name in self.policies:
            return self.policies[file_name]

        temp_path = None

        try:

            # PaddleOCR-VL expects a file path.
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            # ---------------------------------------------
            # OCR
            # ---------------------------------------------
            # -------------------------------------------------
            # PaddleOCR-VL 1.6
            # -------------------------------------------------

           # results = self.ocr.predict(
               # temp_path
           # )
           
            # -------------------------------------------------
            # Extract page-level content
            # -------------------------------------------------
            
           # pages = []
           # full_text_parts = []

            #for page_number, result in enumerate(
             #   results,
              #  start=1,
            #):

             #   markdown = result.markdown or ""

              #  pages.append(
               #     {
                #        "page_number": page_number,
                 #       "text": markdown,
                  #  }
                #)

                #full_text_parts.append(
                 #   markdown
                #)

            #full_text = "\n\n".join(
             #   full_text_parts
            #)

             # With out OCR
            _, page_texts, ocr_pages =  PolicyAnalyzer.extract_pdf_pages(temp_path, "eng")

            # ---------------------------------------------
            # Chunk extracted text
            # ---------------------------------------------

            policy_chunks = PolicyAnalyzer.chunk_policy_pages(file_name, 
                                                       page_texts,
                                                       self.chunk_size,
                                                       self.chunk_overlap)
            # ---------------------------------------------
            # Generate embeddings
            # ---------------------------------------------

         #   embeddings = self.embedding_model.encode(
          #      policy_chunks,
           #     normalize_embeddings=True,
            #)
            embeddings = PolicyAnalyzer.embed_policy_chunks(policy_chunks, self.embedding_model)
            # ---------------------------------------------
            # Store everything in memory
            # ---------------------------------------------

            policy_data = {
                "file_name": file_name,
               # "pages": pages,
                "full_text": page_texts,
                "ocr_pages": ocr_pages,
                "chunks": policy_chunks,
                "embeddings": np.asarray(
                    embeddings,
                    dtype=np.float32,
                ),
            }

            self.policies[file_name] = policy_data

            return policy_data

        finally:

            # Delete temporary PDF
            if temp_path:

                Path(temp_path).unlink(
                    missing_ok=True
                )

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------

    
    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        question: str,
        policy_name: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search over in-memory policy chunks.
        """

        if policy_name:

            if policy_name not in self.policies:
                return []

            policies = {
                policy_name: self.policies[
                    policy_name
                ]
            }

        else:

            policies = self.policies

        if not policies:
            return []

        # ---------------------------------------------
        # Embed user question
        # ---------------------------------------------

        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        results = []

        # ---------------------------------------------
        # Compare against every policy chunk
        # ---------------------------------------------

        for current_policy_name, policy in policies.items():

            embeddings = policy["embeddings"]

            if len(embeddings) == 0:
                continue

            # Because embeddings are normalized,
            # dot product = cosine similarity.
            scores = embeddings @ query_embedding

            top_indices = np.argsort(
                scores
            )[::-1][:top_k]

            for index in top_indices:

                results.append(
                    {
                        "policy_name": current_policy_name,
                        "chunk_index": int(index),
                        "score": float(
                            scores[index]
                        ),
                        "text": policy[
                            "chunks"
                        ][index],
                    }
                )

        # Sort across all policies
        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results[:top_k]


    #sementic serach 

    def relevant_chunks_semantic_search(
        self,
        question: str,
       # policy_name: str | None = None,
        top_k: int = 5,
        ) -> list[dict]:
        
        # If a policy is specified, search only that policy.

       # if policy_name is not None:
        #    if policy_name not in self.policies:
         #       return []

          #  policies_to_search = {
           #     policy_name: self.policies[policy_name]
            #    }
        #else:
        policies_to_search = self.policies

        # Embed the user's question once.
        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
            )

        results = []

        for name, policy in policies_to_search.items():
            chunks = policy["chunks"]
            embeddings = policy["embeddings"]

            if not chunks or len(embeddings) == 0:
                continue

        # Normalized vectors: dot product equals cosine similarity.
            scores = embeddings @ query_embedding

        # Select the highest-scoring chunks.
            k = min(top_k, len(chunks))
            top_indices = np.argsort(scores)[::-1][:k]

            for index in top_indices:
                chunk = chunks[int(index)]

                results.append({
                    "policy_name": name,
                    "chunk": chunk,
                    "score": float(scores[index]),
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "clause_heading": chunk.clause_heading,
                    "text": chunk.text,
                    })

    # When searching multiple policies, rank them together.
        results.sort(
            key=lambda item: item["score"],
            reverse=True,
            )

        return results[:top_k]

    def Analyze_policy(self, policy_name: str) -> dict[str, Any] | None:
        """
        Analyze a policy and return its features.
        """

        if policy_name not in self.policies:
            return None

        policy = self.policies[policy_name]

        for policy in self.policies.values():
            chunks = policy["chunks"]
            features = PolicyAnalyzer.extract_policy_features(chunks)
            policy["features"] = features
       

        return features

    # ---------------------------------------------------------
    # Get policy
    # ---------------------------------------------------------

    def get_policy(
        self,
        policy_name: str,
    ) -> dict[str, Any] | None:

        return self.policies.get(
            policy_name
        )

    # ---------------------------------------------------------
    # Get all policies
    # ---------------------------------------------------------

    def get_policy_names(self) -> list[str]:

        return list(
            self.policies.keys()
        )

    # ---------------------------------------------------------
    # Number of policies
    # ---------------------------------------------------------

    def policy_count(self) -> int:

        return len(
            self.policies
        )

    # ---------------------------------------------------------
    # Remove policy
    # ---------------------------------------------------------

    def remove_policy(
        self,
        policy_name: str,
    ) -> bool:

        if policy_name not in self.policies:
            return False

        del self.policies[
            policy_name
        ]

        return True

    # ---------------------------------------------------------
    # Clear everything
    # ---------------------------------------------------------

    def clear(self):

        self.policies.clear()

    # ---------------------------------------------------------
    # Check if policy exists
    # ---------------------------------------------------------

    def has_policy(
        self,
        policy_name: str,
    ) -> bool:

        return policy_name in self.policies