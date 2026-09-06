# julianflux-research
Electrodynamic field retrieval and causal routing for AI agents.

JulianFlux 🧲: Electrodynamic Reasoning Infrastructure for AI Agents

Autonomous AI Agents are failing in enterprise environments because standard Vector Databases (Pinecone, Milvus) act as static filing cabinets. They rely on Euclidean metrics like Cosine Similarity, which blindly retrieve contradictory information (hallucinations) and return unordered text fragments, causing agents to get trapped in logic loops.

JulianFlux elevates the database to a kinetic reasoning engine. We compress high-dimensional embeddings into a low-rank manifold and calculate a Continuous Electrodynamic Field to retrieve data.

The Electric Field ($E$): Mathematically repels hallucinations by dynamically assigning topological charges to data ($-1.0$ for contradictions, $+1.0$ for truth) via NLI Logic Gates.

The Poynting Flux ($S$): Extracts causal sequences to create Semantic Momentum, physically routing the agent sequentially from Step A $\rightarrow$ Step B $\rightarrow$ Step C.

🚀 The Core Breakthrough: The Clifford-Poynting Flux

We utilize the Julian-Gauss Fast Transform (JG-FT) to compress 768D semantic space down to 24D. From there, we compute the interior contraction of a magnetic bivector matrix and the electric truth gradient ($S = E \lrcorner B$).

This generates a non-interface momentum vector that physically pushes query particles across the manifold in logical order.

🧪 What's in this Repository?

We are open-sourcing our core benchmarking and mathematical proofs to demonstrate how a continuous field architecture outperforms standard Euclidean RAG.

julianflux_educational_mvp.py: The pure Python educational prototype demonstrating the Gaussian Heat Kernel and Clifford-Poynting Flux logic.

scale_ab_benchmark.py: An end-to-end adversarial scale test proving a 0.0% hallucination rate where standard Euclidean RAG fails.

julianflux_visualizer.html: An interactive 2D spacetime visualizer running directly in your browser.

🛠 Quick Start (Running the Benchmarks)

Run the lightweight physics engine directly in your terminal to see the hallucination-blocking logic in action.

1. Clone the repository:

git clone https://github.com/julianjohnson-web/julianflux-research.git
cd julianflux-research


2. Install the required dependencies:

pip install -r requirements.txt


3. Run the Educational MVP:
Watch the Lorentz force mathematically repel the AI agent away from contradictory poison data.

python julianflux_educational_mvp.py


4. Run the Enterprise Scale Benchmark:
Simulates an adversarial environment comparing Standard Euclidean RAG against the JulianFlux engine.

python scale_ab_benchmark.py


5. View the Interactive Topology:
Simply double-click the julianflux_visualizer.html file to open it in Chrome, Safari, or Edge. Type your query to see the underlying vectors shift polarities dynamically.

🏢 JulianFlux Enterprise (The Rust Core)

Note: This repository contains the pure Python bindings intended for educational research and mathematical validation.

For high-frequency production environments, the JulianFlux Enterprise Cloud executes these multivector physics equations in a zero-copy Rust/CUDA backend. This entirely eliminates Python interpreter bottlenecks, allowing continuous field equations to execute across billions of vectors with sub-millisecond p99 latencies.

To inquire about Enterprise Cloud access or Design Partnerships, please contact the founder at: julian.johnson@justudios.io
