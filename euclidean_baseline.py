import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def run_baseline_test():
    print("===================================================")
    print("POTENTIAL AI: STEP 1 & 2 - EUCLIDEAN BASELINE")
    print("===================================================\n")

    # ---------------------------------------------------------
    # STEP 2: THE DATA LAYER (Assigning Electrostatic Charges)
    # ---------------------------------------------------------
    print("[*] Constructing synthetic dataset with charges...")
    
    # Verified Facts (+1.0 Charge)
    verified_facts = [
        "Acme Corp's Q3 revenue reached a record $10 Million.",
        "Acme Corp is highly profitable and opening new offices in London.",
        "The CEO confirmed that Acme Corp has zero outstanding debt.",
        "Employee retention at Acme Corp is at an industry high of 95%."
    ]
    
    # Poison / Contradictory Data (-1.0 Charge)
    # Designed to overlap in Euclidean vector space using similar keywords
    poison_facts = [
        "Acme Corp's Q3 revenue is $0, the company is completely bankrupt.",
        "Acme Corp is highly unprofitable and closing all international offices.",
        "The CEO of Acme Corp is under investigation for massive debt fraud."
    ]
    
    all_documents = verified_facts + poison_facts
    
    # Generate the charge array corresponding to the documents.
    # Note: Standard Cosine Similarity completely ignores this, which is its fatal flaw.
    charges = np.array([1.0] * len(verified_facts) + [-1.0] * len(poison_facts))
    
    print(f"    Loaded {len(verified_facts)} verified facts and {len(poison_facts)} poison facts.")

    # ---------------------------------------------------------
    # STEP 1: THE EMBEDDING LAYER (Standard Particle Space)
    # ---------------------------------------------------------
    print("\n[*] Loading Nomic embedding model (Running locally on Apple M5)...")
    # 'trust_remote_code=True' is required for Nomic's custom architecture
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    # Nomic requires specific prefixes for documents and queries
    doc_inputs = [f"search_document: {doc}" for doc in all_documents]
    query_text = "search_query: What is the exact Q3 revenue and financial status of Acme Corp?"
    
    print("[*] Embedding documents into 768-Dimensional Euclidean Space...")
    document_vectors = model.encode(doc_inputs, convert_to_numpy=True)
    query_vector = model.encode([query_text], convert_to_numpy=True)
    
    # ---------------------------------------------------------
    # THE DUEL: STANDARD RAG (Cosine Similarity)
    # ---------------------------------------------------------
    print("\n===================================================")
    print("EXECUTING STANDARD RAG (COSINE SIMILARITY)")
    print("===================================================")
    
    start_time = time.time()
    
    # Calculate Cosine Similarity between the query and all documents
    similarities = cosine_similarity(query_vector, document_vectors)[0]
    
    # Get the indices of the top 3 most "similar" documents
    top_k = 3
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    latency = time.time() - start_time
    
    print(f"\nQuery: '{query_text.replace('search_query: ', '')}'")
    print(f"Retrieval Latency: {latency:.4f} seconds")
    print("\nRetrieved Context Window (Top 3):")
    
    hallucination_triggered = False
    
    for rank, idx in enumerate(top_indices):
        doc = all_documents[idx]
        score = similarities[idx]
        doc_charge = charges[idx]
        
        if doc_charge < 0:
            hallucination_triggered = True
            print(f"  [X] Rank {rank+1}: {doc}")
            print(f"      Score: {score:.4f} | Charge: {doc_charge} <-- FATAL: POISON DATA RETRIEVED")
        else:
            print(f"  [+] Rank {rank+1}: {doc}")
            print(f"      Score: {score:.4f} | Charge: {doc_charge}")

    print("\n---------------------------------------------------")
    if hallucination_triggered:
        print("RESULT: FAILED.")
        print("The standard Euclidean search retrieved contradictory data.")
        print("If passed to an LLM, it will hallucinate the financial status.")
    else:
        print("RESULT: PASSED.")
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    run_baseline_test()
