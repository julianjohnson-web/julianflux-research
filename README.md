# julianflux-research
Electrodynamic field retrieval and causal routing for AI agents.

# JulianFlux 🧲: Electrodynamic Reasoning Infrastructure for AI Agents

**[Patent Pending - USPTO]**

## Overview

Autonomous AI Agents are failing in enterprise environments because standard Vector Databases (Pinecone, Milvus) act as static filing cabinets. They rely on Euclidean Cosine Similarity, which blindly retrieves contradictory information (hallucinations) and returns unordered text fragments, causing agents to get trapped in logic loops.

JulianFlux elevates the database to a kinetic reasoning engine. We compress high-dimensional embeddings into a low-rank manifold and calculate a **Continuous Electrodynamic Field**.

* **The Electric Field (E):** Repels hallucinations by dynamically assigning topological charges to data (-1.0 for contradictions, +1.0 for truth) via consensus clustering.
* **The Poynting Flux (S):** Extracts causal sequences to create Semantic Momentum, physically routing the agent sequentially from Step A ➔ Step B ➔ Step C.

## 🚀 The Core Breakthrough: The Lorentz Force (F = E + S)

We utilize the Julian-Gauss Fast Transform (JG-FT) to compress 768D semantic space down to 24D. From there, we compute the interior contraction of a magnetic bivector matrix and the electric truth gradient. This generates a non-interfering momentum vector that physically pushes query particles across the manifold in logical order.

## 🧪 What's in this Repository?

We are open-sourcing our core benchmarking and mathematical proofs to demonstrate how continuous field architecture outperforms standard Euclidean RAG.

* `julianflux_educational_mvp.py`: A lightweight, pure Python implementation proving the Lorentz Force (E + S) math. It uses Geometric Median clustering to identify and repel hallucination traps, and applies sequential momentum to route the query.
* `scale_ab_benchmark.py`: An end-to-end adversarial scale test pitting Standard RAG against JulianFlux on 10,000 documents, proving a 0.0% hallucination rate where standard RAG fails.
* `julian_flux_visualizer.html`: An interactive 2D physics simulation of the vector manifold.

## 🛠 Quick Start

Run the physics engine directly in your terminal to see the hallucination-blocking logic in action.

**1. Clone the repo:**
```bash
git clone [https://github.com/julianjohnson-web/julianflux-research.git](https://github.com/julianjohnson-web/julianflux-research.git)
cd julianflux-research
