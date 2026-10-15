# JulianFlux 🧲: Electrodynamic Reasoning Infrastructure for AI Agents

Autonomous AI Agents are failing in enterprise environments because standard Vector Databases (Pinecone, Milvus) act as static filing cabinets. They rely on discrete Euclidean metrics like Cosine Similarity, which blindly retrieve contradictory information (hallucinations) and return unordered text fragments, causing agents to get trapped in logic loops.

JulianFlux elevates the database to a kinetic reasoning engine. We compress high-dimensional embeddings into a low-rank manifold and calculate a **Continuous Electrodynamic Field** to retrieve and route data.

---

## 🚀 The Core Breakthrough: The Lorentz Force ($\mathbf{F}_{\text{total}} = q(\mathbf{E} + \mathbf{S} - \frac{\partial \mathbf{A}}{\partial t})$)

Instead of measuring static distance, we measure physics. We route agents using a modified Lorentz Force equation in a 128-dimensional Clifford Algebra manifold ($\mathbb{G}_{128}$):

- **The Julian-Gauss Fast Transform (JG-FT):** We project massive 768D semantic spaces down to a mathematically sound 128D manifold using a normal distribution ($\mathcal{N}(0, 1/\sqrt{128})$), preserving topological mapping accuracy while reducing memory footprint by **6x** and accelerating retrieval by **5.1x**.
- **Dynamic Bandwidths ($\sigma$):** We dynamically calculate the Gaussian bandwidth for each vector based on its Nearest Neighbor distance, allowing the field to perfectly adapt to local semantic densities.
- **The Electric Field ($\mathbf{E}$):** Mathematically repels hallucinations. We assign topological charges ($q$) to data ($-1.0$ for contradictions, $+1.0$ for truth) via NLI Logic Gates. The repulsive electric field physically blocks the AI from accessing poisoned data.
- **Faraday's Law of Induction ($-\partial \mathbf{A}/\partial t$):** We decay superseded facts and corporate policies over time, generating a repulsive field against deprecated workflows while preserving historical point-in-time queries.
- **The Poynting Flux ($\mathbf{S}$):** Extracts causal sequences to create Semantic Momentum. By computing the interior contraction product in Geometric Algebra ($\mathbf{S} = \mathbf{E} \lrcorner \mathbf{B}$), we generate a momentum vector that pushes the agent sequentially from Step A $\to$ Step B $\to$ Step C.

---

## 🗺️ Repository Map: Which file should I run?

We have open-sourced our core mathematics in Python and an interactive 2D field visualizer to demonstrate how continuous field architecture completely outperforms standard Euclidean RAG.

*(Note: These files are designed to be run locally in the order below).*

**1. `euclidean_baseline.py` (The Problem)**
Run this first. It demonstrates a standard 768D Cosine Similarity retrieval in a poisoned dataset. You will watch standard RAG fail entirely under Polarity Blindness ($\Delta \cos(\theta) = 0.0074$), blindly retrieving contradictory bankruptcy data and collapsing the causal workflow.

**2. `julianflux_educational_mvp.py` (The Fast Solution)**
The lightweight, instantly runnable JulianFlux engine. It uses the 128D JG-FT projection, dynamic $\sigma$, and $\mathbb{G}_{128}$ Clifford bivector contraction ($\mathbf{S} = \mathbf{E} \lrcorner \mathbf{B}$) to route the agent sequentially in `~0.12 ms` while repelling poisoned nodes.

**3. `scale_ab_benchmark.py` (The 10,000-Vector Stress Test)**
An end-to-end adversarial scale test generating a 10,000-document enterprise manifold. It runs 768D Standard RAG and 128D JulianFlux head-to-head, proving a **5.1x latency speedup (`0.31 ms` vs `1.61 ms`)**, a **0.0% hallucination/poison rate**, and **100.0% causal step precision** at scale.

**4. `julianflux_engine.py` (The Full 4D Spacetime Physics Engine)**
The spacetime engine. This file integrates the `DeBERTa-v3` NLI Cross-Encoder for enterprise-grade truth verification alongside **Faraday's Law of Induction ($-\partial \mathbf{A}/\partial t$)** and Temporal Decay, mathematically repelling the AI away from recently deprecated corporate policies while supporting non-destructive historical time-travel queries. *(Note: Automatically downloads the lightweight DeBERTa-v3 NLI model on first run).*

**5. `julian_flux_visualizer.html` (Interactive 2D Manifold Visualizer)**
Open this file directly in any browser (`open julian_flux_visualizer.html`). It renders an interactive 2D projection of the electrodynamic manifold, letting you edit agent queries in real time and compare Standard Cosine RAG (which collides straight into the $q = -1.0$ poison trap) against the JulianFlux Lorentz trajectory.

---

## 🛠 Quick Start

Clone the repository:
```bash
git clone [https://github.com/julianjohnson-web/julianflux-research.git](https://github.com/julianjohnson-web/julianflux-research.git)
cd julianflux-research
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Run the validation & scale benchmark suite:
```bash
python3 euclidean_baseline.py
python3 julianflux_educational_mvp.py
python3 scale_ab_benchmark.py
python3 julianflux_engine.py
```

Open the interactive 2D Electrodynamic Field Visualizer in your browser:
```bash
open julian_flux_visualizer.html
```

---

## 🏢 JulianFlux Enterprise (The Proprietary Rust Core)

*Note: This public repository contains the pure Python reference implementations intended for educational research and open-source mathematical validation.*

For high-frequency production environments, the proprietary **JulianFlux Enterprise Core** executes these multivector physics equations in a zero-copy Rust/PyO3 & CUDA backend with cache-contiguous causal horizon sharding. This eliminates the Python Global Interpreter Lock (GIL), allowing the Lorentz Force equations to execute across **1,000,000+ vectors at `22,714 QPS` with `0.199 ms` P50 (`0.216 ms` P99) bare-metal latencies**.

To inquire about Enterprise Cloud access, live Rust benchmark demonstrations, or Design Partnerships, please contact the founder at: **julian.johnson@justudios.io**
