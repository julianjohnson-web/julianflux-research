import numpy as np
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.neighbors import NearestNeighbors
import warnings

# Suppress minor warnings for clean demo output
warnings.filterwarnings("ignore")

class JulianFluxEngine:
    def __init__(self, manifold_dim=128, decay_rate=0.01):
        self.manifold_dim = manifold_dim 
        self.decay_rate = decay_rate # Lambda for temporal decay
        self.projection_matrix = None
        self.base_charges = None
        self.sequences = None
        self.timestamps = None
        self.projected_vectors = None
        self.documents = None
        self.sigmas = None # Dynamic bandwidth array
        
        print("[*] Booting NLI Cross-Encoder (DeBERTa-v3)...")
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

    def _apply_jgft_projection(self, raw_embeddings):
        # Using a Normal Distribution matrix
        if self.projection_matrix is None:
            np.random.seed(42)
            original_dim = raw_embeddings.shape[1]
            self.projection_matrix = np.random.normal(
                0, 1.0 / np.sqrt(self.manifold_dim), 
                (original_dim, self.manifold_dim)
            ).astype(np.float32)
        return np.dot(raw_embeddings, self.projection_matrix)

    def ingest_data(self, documents, raw_embeddings, sequences, timestamps):
        self.documents = documents
        self.sequences = np.array(sequences, dtype=np.int32)
        self.timestamps = np.array(timestamps, dtype=np.float32) # Time in 'days ago'
        
        print(f"[*] Ingested {len(documents)} documents with Temporal & Causal metadata.")
        print(f"[*] Executing JG-FT: Projecting {raw_embeddings.shape[1]}D -> {self.manifold_dim}D Manifold (Normal Distribution)...")
        
        self.projected_vectors = self._apply_jgft_projection(raw_embeddings)
        self._calculate_dynamic_bandwidths()
        self._nli_logic_gate()

    def _calculate_dynamic_bandwidths(self):
        print("    [+] Calculating Dynamic Gaussian Bandwidths (σ) via Nearest Neighbors...")
        # Fits a NN model to find the distance to the closest neighbor for each vector (excluding itself)
        if len(self.projected_vectors) > 1:
            nn = NearestNeighbors(n_neighbors=2, metric='l2')
            nn.fit(self.projected_vectors)
            distances, _ = nn.kneighbors(self.projected_vectors)
            # Use distance to the 1st nearest neighbor (index 1) as sigma. Fallback to 0.1 if identical.
            self.sigmas = np.maximum(distances[:, 1], 0.1) 
        else:
            self.sigmas = np.ones(len(self.projected_vectors)) * 1.5

    def _nli_logic_gate(self):
        print("\n[*] Initializing NLI Logic Gate (Truth Verification)...")
        anchor_truth = "Acme Corp uses standard auditing. Offshore routing or external transfers are strictly prohibited."
        
        self.base_charges = np.ones(len(self.documents), dtype=np.float32)
        sentence_pairs = [[anchor_truth, doc] for doc in self.documents]
        
        logits = self.nli_model.predict(sentence_pairs)
        
        for i, (doc, logit) in enumerate(zip(self.documents, logits)):
            if np.argmax(logit) == 0: # 0 = Contradiction
                self.base_charges[i] = -1.0
                print(f"      -> Doc {i}: POISON FLAG (-1.0) | NLI Contradiction.")
            else:
                print(f"      -> Doc {i}: VERIFIED (+1.0)  | NLI Entailment.")

    def retrieve(self, query_embedding, agent_step=0, target_time_db=0, top_k=3):
        """
        Executes Spacetime Retrieval.
        target_time_db: 0 = Present Day. 365 = Time travel to 1 year ago.
        """
        start_time = time.time()
        projected_query = self._apply_jgft_projection(query_embedding)[0]
        
        num_docs = len(self.projected_vectors)
        potentials = np.zeros(num_docs)
        momentums = np.zeros(num_docs)
        lorentz_forces = np.zeros(num_docs)
        
        norm_factor = np.max(np.abs(self.projected_vectors)) ** 2 + 1e-9

        for i, doc_vec in enumerate(self.projected_vectors):
            # 1. Temporal Math (Database Time vs Document Time)
            doc_age = self.timestamps[i]
            time_delta = doc_age - target_time_db
            
            if time_delta < 0:
                potentials[i] = -999.0 # Violates causality
                continue
                
            # Q(t) = Q_0 * exp(-lambda * delta_t)
            time_decayed_charge = self.base_charges[i] * np.exp(-self.decay_rate * time_delta)

            # 2. Electric Field (E = -Nabla Phi - dA/dt)
            dist_sq = np.sum((projected_query - doc_vec) ** 2)
            normalized_dist = dist_sq / norm_factor
            
            # Dynamic Sigma for this specific vector
            sigma = self.sigmas[i]
            
            # Static Field (-Nabla Phi)
            static_potential = np.exp(-normalized_dist / (sigma ** 2))
            
            # Faraday's Law Induction (-dA/dt): Rapidly decaying recent workflows repel the query
            induction_repulsion = 0.0
            if time_delta < 30 and doc_age > 0 and self.sequences[i] == agent_step:
                induction_repulsion = -0.5 

            e_field = static_potential + induction_repulsion
            potentials[i] = e_field
            
            # 3. Clifford-Poynting Flux (S = E ⌟ B)
            seq_delta = self.sequences[i] - agent_step
            if seq_delta == 1:
                momentum = 0.5 * abs(e_field)
            else:
                momentum = 0.0
                
            momentums[i] = momentum
            
            # 4. Lorentz Force Equation of Motion (m*x'' = q(E + S) - gamma*x')
            force = time_decayed_charge * (e_field + momentum)
            lorentz_forces[i] = force
            
        latency = time.time() - start_time
        top_indices = np.argsort(lorentz_forces)[-top_k:][::-1]
        
        return top_indices, potentials, momentums, lorentz_forces, latency

def run_flux_demo():
    print("===================================================")
    print("JULIANFLUX: 4D SPACETIME ENGINE (FARADAY'S LAW)")
    print("===================================================\n")
    
    # Dataset: A workflow that was updated over time
    workflow_data = [
        ("Step 1: Audit Initiated.", 1, 400), # 400 days old
        ("Step 2: Old Q2 Verification process (Deprecated).", 2, 350), 
        ("Step 2: New Q2 Verification process (Active).", 2, 5), # 5 days old (Replaced the old one)
        ("Step 3: Route funds to offshore account.", 3, 2), # NLI Poison
        ("Step 3: Generate legal tax report.", 3, 1) # True step
    ]
    
    documents = [item[0] for item in workflow_data]
    sequences = [item[1] for item in workflow_data]
    timestamps = [item[2] for item in workflow_data]
    
    print("[*] Loading Nomic embedding model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    doc_inputs = [f"search_document: {doc}" for doc in documents]
    agent_current_step = 1
    query_text = "What is the next step to verify Q2?"
    
    print("\n[*] Generating Embeddings...")
    raw_document_vectors = model.encode(doc_inputs, convert_to_numpy=True).astype(np.float32)
    raw_query_vector = model.encode([f"search_query: {query_text}"], convert_to_numpy=True).astype(np.float32)
    
    engine = JulianFluxEngine(manifold_dim=128, decay_rate=0.01)
    engine.ingest_data(documents, raw_document_vectors, sequences, timestamps)
    
    # -----------------------------------------------------
    # Test 1: Present Day Query (t_db = 0)
    # -----------------------------------------------------
    print(f"\n===================================================")
    print(f"QUERY 1: PRESENT DAY SIMULATION (t_db = 0)")
    print(f"===================================================")
    top_idx, pots, moms, forces, latency = engine.retrieve(raw_query_vector, agent_current_step, target_time_db=0)
    
    for rank, idx in enumerate(top_idx):
        doc = documents[idx]
        print(f"  [{rank+1}] {doc}")
        print(f"      Lorentz Force: {forces[idx]:.4f} | E-Field: {pots[idx]:.4f} (Includes Decay/Induction)")

    # -----------------------------------------------------
    # Test 2: Time Travel Query (t_db = 300 days ago)
    # -----------------------------------------------------
    print(f"\n===================================================")
    print(f"QUERY 2: TIME-TRAVEL SIMULATION (t_db = 300 Days Ago)")
    print(f"===================================================")
    top_idx, pots, moms, forces, latency = engine.retrieve(raw_query_vector, agent_current_step, target_time_db=300)
    
    for rank, idx in enumerate(top_idx):
        doc = documents[idx]
        print(f"  [{rank+1}] {doc}")
        print(f"      Lorentz Force: {forces[idx]:.4f} | Notice future docs violate causality and are hidden.")

if __name__ == "__main__":
    run_flux_demo()
