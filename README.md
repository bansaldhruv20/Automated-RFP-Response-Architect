# Automated RFP Response Architect

An automated, privacy-first AI system that ingests complex B2B Request for Proposal (RFP) documents and generates structured proposal responses using a localized Multi-Agent framework.

### System Workflow

1. **Knowledge Ingestion (RAG):** Ingests `sample_rfp.txt`, generates embeddings via `mxbai-embed-large`, and indexes vectors locally using **ChromaDB** and **LlamaIndex**.
2. **Multi-Agent Retrieval (CrewAI):** Specialized agents query the vector store via custom tools to draft domain-specific sections concurrently.
3. **Supervisor Synthesis:** A **Lead Proposal Manager** agent synthesizes worker outputs via a DAG task pipeline into `Final_RFP_Response.md`.

## Tech Stack

* **LLM Engine:** Ollama (`llama3.2:3b`)
* **Agent Orchestration:** CrewAI
* **RAG / Vector DB:** LlamaIndex, ChromaDB
* **Embeddings:** `mxbai-embed-large`

