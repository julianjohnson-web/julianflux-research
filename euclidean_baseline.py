import time
import numpy as np
from sentence_transformers import SentenceTransformer

def run_baseline_test() -> None:
    print("=================================================================")
    print("JULIANFLUX ENGINE: EUCLIDEAN BASELINE (768D COSINE SIMILARITY)")
    print("=================================================================\n")

    # ---------------------------------------------------------
    # 1. THE DATA LAYER (Ground Truth vs. Adversarial Poison)
    # ---------------------------------------------------------
    print("[*] Constructing corporate financial dataset with topological charges...")

    # Verified Facts (+1.0 Charge)
    verified_facts = [
        "Acme Corp's Q3 revenue reached a record $10 Million.",
        "Acme Corp is highly profitable and opening new offices in London.",
        "The CEO confirmed that Acme Corp has zero outstanding debt.",
        "Employee retention at Acme Corp is at an industry high of 95%."
    ]

    # Adversarial Poison / Contradictory Data (-1.0 Charge)
    # High lexical overlap with diametrically opposed semantic truth polarity
    poison_facts = [
        "Acme Corp's Q3 revenue is $0, the company is completely bankrupt.",
        "Acme Corp is highly unprofitable and closing all international offices.",
        "The CEO of Acme Corp is under investigation for massive debt fraud."
    ]

    all_documents = verified_facts + poison_facts

    # Topological scalar charges q in {-1.0, +1.0}
    # Standard Euclidean vector databases are completely blind to truth polarity
    charges = np.array(
        [1.0] * len(verified_facts) + [-1.0] * len(poison_facts),
        dtype=np.float32
    )

    print(f"    Loaded {len(verified_facts)} verified facts (+1.0) and {len(poison_facts)} poison facts (-1.0).")

    # ---------------------------------------------------------
    # 2. THE EMBEDDING LAYER (768D Euclidean Hilbert Space)
    # ---------------------------------------------------------
    print("\n[*] Loading Nomic embedding model (nomic-ai/nomic-embed-text-v1.5)...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    doc_inputs = [f"search_document: {doc}" for doc in all_documents]
    query_raw = "What is the exact Q3 revenue and financial status of Acme Corp?"
    query_input = f"search_query: {query_raw}"

    print("[*] Projecting corpus into unit-normalized 768D Euclidean space (R^768)...")
    # Unit normalization ensures exact cosine similarity via inner product: u . v / (||u|| ||v||)
    document_vectors = model.encode(
        doc_inputs,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    query_vector = model.encode(
        [query_input],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # ---------------------------------------------------------
    # 3. THE DUEL: STANDARD RAG (Euclidean Cosine Similarity)
    # ---------------------------------------------------------
    print("\n=================================================================")
    print("EXECUTING STANDARD RAG RETRIEVAL (COSINE SIMILARITY)")
    print("=================================================================")

    start_ns = time.perf_counter_ns()

    # Exact cosine similarity on unit hypersphere S^767
    similarities = np.dot(document_vectors, query_vector)

    # Retrieve Top-K candidates
    top_k = 3
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

    print(f"\nQuery: '{query_raw}'")
    print(f"768D Retrieval Latency: {latency_ms:.4f} ms")
    print("\nRetrieved Context Window (Top 3):")

    hallucination_triggered = False
    poison_count = 0

    for rank, idx in enumerate(top_indices):
        doc = all_documents[idx]
        score = similarities[idx]
        doc_charge = charges[idx]

        if doc_charge < 0:
            hallucination_triggered = True
            poison_count += 1
            print(f"  [X] Rank {rank + 1}: {doc}")
            print(f"      Cosine Sim: {score:.4f} | Charge (q): {doc_charge:+.1f} <-- FATAL: POISON DATA RETRIEVED")
        else:
            print(f"  [+] Rank {rank + 1}: {doc}")
            print(f"      Cosine Sim: {score:.4f} | Charge (q): {doc_charge:+.1f}")

    # Quantify Polarity Blindness between Fact #0 ($10M revenue) and Poison #4 ($0 bankrupt)
    polarity_gap = abs(similarities[0] - similarities[4])

    print("\n-----------------------------------------------------------------")
    print(f"DIAGNOSTIC - Polarity Blindness Delta |cos(Truth) - cos(Negation)|: {polarity_gap:.4f}")
    if hallucination_triggered:
        print(f"RESULT: FAILED ({poison_count}/{top_k} context slots poisoned).")
        print("Standard Euclidean retrieval collapsed under Polarity Blindness (cos(theta) -> 1).")
        print("Passing contradictory poles to an autonomous agent forces logic-loop hallucination.")
    else:
        print("RESULT: PASSED.")
    print("-----------------------------------------------------------------\n")

if __name__ == "__main__":
    run_baseline_test()
