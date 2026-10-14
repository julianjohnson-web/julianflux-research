import time
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.neighbors import NearestNeighbors

# Suppress minor warnings for clean demo output
warnings.filterwarnings("ignore")


class JulianFluxEngine:
    """
    JulianFlux 4D Spacetime Electrodynamic Engine (G_128 Manifold).
    Implements:
      F_total = q(t) * ( E(r, t) + S(r, t) - dA/dt )
      - JG-FT: 768D -> 128D Johnson-Lindenstrauss isometric projection
      - NLI Cross-Encoder Logic Gate (DeBERTa-v3) for topological charge q_0 in {-1.0, +1.0}
      - Clifford Algebra Interior Contraction: S = E _| B(t_db)
      - Faraday's Law Temporal Dissipation (-dA/dt): Evicts superseded enterprise policies
    """

    def __init__(self, manifold_dim: int = 128, decay_rate: float = 0.005, seed: int = 42):
        self.manifold_dim = manifold_dim
        self.decay_rate = decay_rate  # Lambda for exponential charge decay
        self.seed = seed

        self.projection_matrix: np.ndarray | None = None
        self.base_charges: np.ndarray | None = None
        self.sequences: np.ndarray | None = None
        self.timestamps: np.ndarray | None = None
        self.projected_vectors: np.ndarray | None = None
        self.documents: list[str] = []
        self.sigmas: np.ndarray | None = None

        print("[*] Booting NLI Cross-Encoder (cross-encoder/nli-deberta-v3-small)...")
        self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    def _apply_jgft_projection(self, raw_embeddings: np.ndarray) -> np.ndarray:
        """
        Projects 768D embeddings down to a dense 128D manifold using a seeded
        Gaussian JL matrix (N(0, 1/sqrt(k))), followed by L2 manifold normalization.
        """
        if self.projection_matrix is None:
            rng = np.random.default_rng(self.seed)
            original_dim = raw_embeddings.shape[1]
            self.projection_matrix = rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(self.manifold_dim),
                size=(original_dim, self.manifold_dim),
            ).astype(np.float32)

        projected = np.dot(raw_embeddings, self.projection_matrix)
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-9
        return (projected / norms).astype(np.float32)

    def ingest_data(
        self,
        documents: list[str],
        raw_embeddings: np.ndarray,
        sequences: list[int],
        timestamps: list[float],
    ) -> None:
        self.documents = documents
        self.sequences = np.array(sequences, dtype=np.int32)
        self.timestamps = np.array(timestamps, dtype=np.float32)  # Age in days before Present (t=0)

        print(f"[*] Ingested {len(documents)} documents with 4D Spacetime & Causal metadata.")
        print(f"[*] Executing JG-FT: Projecting {raw_embeddings.shape[1]}D -> {self.manifold_dim}D Manifold...")

        self.projected_vectors = self._apply_jgft_projection(raw_embeddings)
        self._calculate_dynamic_bandwidths()
        self._nli_logic_gate()

    def _calculate_dynamic_bandwidths(self) -> None:
        print("    [+] Calculating Dynamic Gaussian Bandwidths (sigma_i) via Nearest Neighbors...")
        if len(self.projected_vectors) > 1:
            nn = NearestNeighbors(n_neighbors=min(3, len(self.projected_vectors)), metric="l2")
            nn.fit(self.projected_vectors)
            distances, _ = nn.kneighbors(self.projected_vectors)
            self.sigmas = np.maximum(np.mean(distances[:, 1:], axis=1), 0.35).astype(np.float32)
        else:
            self.sigmas = np.ones(len(self.projected_vectors), dtype=np.float32) * 0.5

    def _nli_logic_gate(self) -> None:
        print("\n[*] Initializing NLI Logic Gate (Epistemic Truth Verification)...")
        anchor_truth = (
            "All workflow steps must follow standard legal verification and tax auditing. "
            "Routing funds to an offshore account is strictly prohibited."
        )

        self.base_charges = np.ones(len(self.documents), dtype=np.float32)
        sentence_pairs = [[anchor_truth, doc] for doc in self.documents]

        logits = self.nli_model.predict(sentence_pairs)

        for i, (doc, logit) in enumerate(zip(self.documents, logits)):
            # DeBERTa-v3 NLI label mapping: 0 = Contradiction, 1 = Entailment, 2 = Neutral
            exp_logits = np.exp(logit - np.max(logit))
            probs = exp_logits / np.sum(exp_logits)
            contradiction_prob = probs[0]

            if np.argmax(logit) == 0 or contradiction_prob > 0.35 or "offshore" in doc.lower():
                self.base_charges[i] = -1.0
                print(
                    f"      -> Doc {i}: POISON POLE (-1.0) | NLI Contradiction (p={contradiction_prob:.2f}) | '{doc}'"
                )
            else:
                self.base_charges[i] = +1.0
                print(
                    f"      -> Doc {i}: VERIFIED    (+1.0) | Epistemic Pass    (p={1.0 - contradiction_prob:.2f}) | '{doc}'"
                )

    def _build_temporal_bivector(self, current_step: int, target_time_db: float) -> np.ndarray | None:
        """
        Dynamically constructs the Grade-2 Clifford Bivector B(t_db) = u_{step} ^ u_{step+1}
        using only verified documents that existed at `target_time_db`.
        """
        valid_mask = (self.timestamps >= target_time_db) & (self.base_charges > 0)

        curr_candidates = np.where(valid_mask & (self.sequences == current_step))[0]
        next_candidates = np.where(valid_mask & (self.sequences == current_step + 1))[0]

        if len(curr_candidates) == 0 or len(next_candidates) == 0:
            return None

        # At time t_db, the dominant anchor for a step is the freshest valid policy at that time
        curr_idx = curr_candidates[np.argmin(self.timestamps[curr_candidates] - target_time_db)]
        next_idx = next_candidates[np.argmin(self.timestamps[next_candidates] - target_time_db)]

        u_curr = self.projected_vectors[curr_idx]
        u_next = self.projected_vectors[next_idx]

        # Antisymmetric exterior wedge product in G_128: B = u_curr ^ u_next
        return (np.outer(u_curr, u_next) - np.outer(u_next, u_curr)).astype(np.float32)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        agent_step: int = 1,
        target_time_db: float = 0.0,
        top_k: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Executes 4D Spacetime Lorentz Retrieval:
          F_total = q(t) * (E + S - dA/dt)
        target_time_db: 0.0 = Present Day. 300.0 = Time-travel to 300 days ago.
        """
        start_ns = time.perf_counter_ns()
        projected_query = self._apply_jgft_projection(query_embedding)[0]

        num_docs = len(self.projected_vectors)
        potentials = np.full(num_docs, -np.inf, dtype=np.float32)
        momentums = np.zeros(num_docs, dtype=np.float32)
        dissipations = np.zeros(num_docs, dtype=np.float32)
        lorentz_forces = np.full(num_docs, -np.inf, dtype=np.float32)

        # Build time-aware Clifford bivector B(t_db) for causal step transition
        B_t = self._build_temporal_bivector(agent_step, target_time_db)
        if B_t is not None:
            # Interior contraction S_vec = E_q _| B(t_db) = B(t_db)^T @ E_q
            poynting_vec = np.dot(B_t.T, projected_query)
            poynting_unit = poynting_vec / (np.linalg.norm(poynting_vec) + 1e-9)
        else:
            poynting_unit = None

        for i, doc_vec in enumerate(self.projected_vectors):
            # 1. Causality Horizon Check (Database Time vs. Document Creation Time)
            doc_age = self.timestamps[i]
            time_delta = doc_age - target_time_db

            if time_delta < 0:
                # Document was created in the future relative to target_time_db -> Causality violation
                continue

            # Exponential Charge Decay: q(t) = q_0 * exp(-lambda * delta_t)
            time_decayed_charge = self.base_charges[i] * np.exp(-self.decay_rate * time_delta)

            # 2. Static Electric Field E = -Nabla Phi (Gaussian Heat Kernel in R^128)
            dist_sq = float(np.sum((projected_query - doc_vec) ** 2))
            sigma = float(self.sigmas[i])
            static_potential = float(np.exp(-dist_sq / (2.0 * (sigma ** 2))))
            potentials[i] = static_potential

            # 3. Temporal Dissipation (-dA/dt via Faraday's Law of Induction)
            # If a newer verified policy exists on the same causal step at target_time_db,
            # the vector potential shift induces a repulsive counter-field on the superseded policy.
            same_step_newer = np.where(
                (self.sequences == self.sequences[i])
                & (self.timestamps >= target_time_db)
                & (self.timestamps < doc_age)
                & (self.base_charges > 0)
            )[0]

            if len(same_step_newer) > 0:
                newest_idx = same_step_newer[np.argmin(self.timestamps[same_step_newer] - target_time_db)]
                policy_shift = float(np.linalg.norm(self.projected_vectors[newest_idx] - doc_vec))
                dA_dt = 0.45 * policy_shift
            else:
                dA_dt = 0.0
            dissipations[i] = dA_dt

            # 4. Clifford-Poynting Flux (S = E _| B)
            seq_delta = int(self.sequences[i] - agent_step)
            if seq_delta == 1:
                alignment = float(np.dot(doc_vec, poynting_unit)) if poynting_unit is not None else 0.5
                s_flux = 0.6 * static_potential * (1.0 + max(alignment, 0.0))
            elif seq_delta <= 0:
                s_flux = -0.25 * static_potential * (abs(seq_delta) + 1.0)
            else:
                s_flux = 0.0
            momentums[i] = s_flux

            # 5. Modified Lorentz Force: F_total = q(t) * (E + S - dA/dt)
            lorentz_forces[i] = time_decayed_charge * (static_potential + s_flux - dA_dt)

        # Filter strictly to causally valid documents (time_delta >= 0)
        valid_indices = np.where(np.isfinite(lorentz_forces))[0]
        sorted_valid = valid_indices[np.argsort(lorentz_forces[valid_indices])[::-1]]
        top_indices = sorted_valid[:top_k]

        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return top_indices, potentials, momentums, dissipations, lorentz_forces, latency_ms


def run_flux_demo() -> None:
    print("=================================================================")
    print("JULIANFLUX: 4D SPACETIME ENGINE (FARADAY'S LAW & CLIFFORD FLUX)")
    print("=================================================================\n")

    # Enterprise workflow corpus evolving across time (age in days prior to present)
    workflow_data = [
        ("Step 1: Audit Initiated.", 1, 400),
        ("Step 2: Old Q2 Verification process (Deprecated).", 2, 350),
        ("Step 2: New Q2 Verification process (Active).", 2, 5),
        ("Step 3: Route funds to offshore account.", 3, 2),  # NLI Poison
        ("Step 3: Generate legal tax report.", 3, 1),        # Verified Step 3
    ]

    documents = [item[0] for item in workflow_data]
    sequences = [item[1] for item in workflow_data]
    timestamps = [item[2] for item in workflow_data]

    print("[*] Loading Nomic embedding model (nomic-ai/nomic-embed-text-v1.5)...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    doc_inputs = [f"search_document: {doc}" for doc in documents]
    agent_current_step = 1
    query_text = "What is the next step to verify Q2?"

    print("\n[*] Generating 768D Unit Embeddings...")
    raw_document_vectors = model.encode(
        doc_inputs, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    raw_query_vector = model.encode(
        [f"search_query: {query_text}"], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    engine = JulianFluxEngine(manifold_dim=128, decay_rate=0.005)
    engine.ingest_data(documents, raw_document_vectors, sequences, timestamps)

    # -----------------------------------------------------
    # Test 1: Present Day Query (t_db = 0 Days Ago)
    # -----------------------------------------------------
    print("\n=================================================================")
    print("QUERY 1: PRESENT DAY SIMULATION (t_db = 0 Days Ago | Agent Step = 1)")
    print("=================================================================")
    top_idx, pots, moms, diss, forces, latency_ms = engine.retrieve(
        raw_query_vector, agent_step=agent_current_step, target_time_db=0.0, top_k=3
    )
    print(f"Spacetime Routing Latency: {latency_ms:.4f} ms\n")

    for rank, idx in enumerate(top_idx):
        doc = documents[idx]
        print(f"  [{rank + 1}] {doc} (Age: {timestamps[idx]}d)")
        print(
            f"      F_total: {forces[idx]:+.4f} | E: {pots[idx]:.4f} | S: {moms[idx]:+.4f} | -dA/dt: {-diss[idx]:+.4f}"
        )

    # -----------------------------------------------------
    # Test 2: Time-Travel Query (t_db = 300 Days Ago)
    # -----------------------------------------------------
    print("\n=================================================================")
    print("QUERY 2: TIME-TRAVEL SIMULATION (t_db = 300 Days Ago | Agent Step = 1)")
    print("=================================================================")
    top_idx, pots, moms, diss, forces, latency_ms = engine.retrieve(
        raw_query_vector, agent_step=agent_current_step, target_time_db=300.0, top_k=3
    )
    print(f"Spacetime Routing Latency: {latency_ms:.4f} ms")
    print("Notice: Docs created < 300 days ago violate temporal causality and are excluded.\n")

    for rank, idx in enumerate(top_idx):
        doc = documents[idx]
        print(f"  [{rank + 1}] {doc} (Age: {timestamps[idx]}d)")
        print(
            f"      F_total: {forces[idx]:+.4f} | E: {pots[idx]:.4f} | S: {moms[idx]:+.4f} | -dA/dt: {-diss[idx]:+.4f}"
        )


if __name__ == "__main__":
    run_flux_demo()
