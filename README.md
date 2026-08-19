# Automated RFP Response Architect

An automated, privacy-first AI system that ingests complex B2B Request for Proposal (RFP) documents and generates structured proposal responses using a localized Multi-Agent framework.

## Architecture Overview
[sample_rfp.txt] ➔ [mxbai-embed-large] ➔ [ChromaDB Index]
│
┌──────────────────────────────────────┴──────────────────────────────────────┐
▼                                                                             ▼
[Security Architect Agent]                                              [Cloud Systems Engineer Agent]
(Drafts Security Section)                                              (Drafts Technical Section)
│                                                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
▼
[Lead Proposal Manager Agent]
│
▼
[Final_RFP_Response.md]

### System Workflow

1. **Knowledge Ingestion (RAG):** Ingests `sample_rfp.txt`, generates embeddings via `mxbai-embed-large`, and indexes vectors locally using **ChromaDB** and **LlamaIndex**.
2. **Multi-Agent Retrieval (CrewAI):** Specialized agents query the vector store via custom tools to draft domain-specific sections concurrently.
3. **Supervisor Synthesis:** A **Lead Proposal Manager** agent synthesizes worker outputs via a DAG task pipeline into `Final_RFP_Response.md`.

## Tech Stack

* **LLM Engine:** Ollama (`llama3.2:3b`)
* **Agent Orchestration:** CrewAI
* **RAG / Vector DB:** LlamaIndex, ChromaDB
* **Embeddings:** `mxbai-embed-large`

## Quickstart

```bash
---

<ElicitationsGroup message="What would you like to do next?">
<Elicitation label="Draft resume bullet points for this RFP project" query="Draft impactful resume bullet points for this Multi-Agent RFP project" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Generate a sample main.py script for this CrewAI workflow" query="Generate a sample main.py script for this CrewAI and LlamaIndex RFP workflow" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Add interview preparation questions for this architecture" query="Give me mock interview questions and answers for this CrewAI RAG project" query_intent="CLICKABLE_SUGGESTION" />
</ElicitationsGroup>
# Pull local models via Ollama
ollama pull llama3.2:3b
ollama pull mxbai-embed-large

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
