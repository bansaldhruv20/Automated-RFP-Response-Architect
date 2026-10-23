# Automated RFP Response Architect

A local AI prototype that retrieves requirements from a text RFP and
produces a proposal draft using three CrewAI agents.

Human review is required before using any generated response.

## Architecture

1. LlamaIndex splits the RFP into overlapping chunks.
2. Hugging Face `BAAI/bge-small-en-v1.5` creates embeddings locally.
3. An in-memory vector index retrieves evidence for security,
   architecture and commercial requirements.
4. A Security Architect and Cloud Systems Engineer draft their sections.
5. A Lead Proposal Manager combines both drafts and the commercial
   evidence into a final proposal.

All three agents use Ollama `llama3.2:3b`. Execution is sequential.

Retrieval happens before agent execution, making the evidence inspectable
and avoiding reliance on small-model tool calling.

This version uses an in-memory LlamaIndex vector index. It does not use
ChromaDB, persistent indexing or concurrent agent execution.

## Requirements

- Python 3.12
- Ollama: https://ollama.com/download
- Several GB of available disk space
- Internet access for initial dependency and model downloads

Inference runs locally when using the default local Ollama endpoint.
No cloud LLM API key is required.

## Installation

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For CPU-only Linux, install CPU PyTorch first to avoid unnecessary GPU
packages:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For the exact dependency set validated on Linux CPU with Python 3.12:

```bash
python -m pip install -r requirements-linux-cpu.lock --extra-index-url https://download.pytorch.org/whl/cpu
```

The lock file is platform-specific. Use `requirements.txt` on other
platforms.

## Start Ollama

If the Ollama desktop application is not already running, start the
server in another terminal:

```bash
ollama serve
```

Download the language model:

```bash
ollama pull llama3.2:3b
```

The embedding model downloads automatically on the first retrieval run.

## Run

Check the input file and Ollama model:

```bash
python main.py --check
```

Test real embeddings and retrieval without generating a proposal:

```bash
python main.py --retrieve-only
```

Generate a proposal:

```bash
python main.py
```

Show detailed agent execution:

```bash
python main.py --verbose
```

Use your own RFP:

```bash
python main.py --input /path/to/rfp.txt --output outputs --timeout 600
```

Only UTF-8 `.txt` RFPs are supported.

Default input and output paths work regardless of the current working
directory.

CPU inference can take several minutes.

## Output files

Each successful run creates a separate `outputs/run-*` directory.

Full proposal runs produce:

- `evidence.json`: retrieved excerpts, source labels and similarity scores.
- `raw_proposal.md`: original model output.
- `Final_RFP_Response.md`: draft with citation-audit warnings.
- `audit.json`: unknown citation labels and review status.
- `run.json`: model settings and measured execution times.

Retrieval-only runs produce `evidence.json` and `run.json`.

The original root-level `Final_RFP_Response.md` is a historical example
and is never overwritten.

## Citation auditing

The application compares generated chunk citations against the
retrieved evidence.

References to nonexistent chunks are marked `UNVERIFIED CITATION`.
They are not silently reassigned to another source.

This checks whether a reference exists. It does not establish that the
source supports the claim, that every requirement was extracted, or
that a proposed vendor capability is true.

## Configuration

Command-line options:

- `--input`
- `--output`
- `--model`
- `--ollama-url`
- `--embedding-model`
- `--timeout`
- `--check`
- `--retrieve-only`
- `--verbose`

Environment defaults:

- `RFP_MODEL`
- `RFP_OLLAMA_URL`
- `RFP_EMBEDDING_MODEL`

The default Ollama endpoint is `http://localhost:11434`.
Using a remote endpoint sends inference prompts to that endpoint.

CrewAI/OpenTelemetry telemetry is disabled by default unless explicitly
overridden in the process environment.

## Tests

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

The 11 automated tests cover:

- Importing without model initialization.
- Default input paths.
- Invalid input handling.
- Ollama readiness and missing-model errors.
- Separate output directories.
- Evidence and timing persistence.
- Three-agent configuration and task dependencies.
- Real CrewAI orchestration with simulated LLM responses.
- Unknown and missing citation handling.

These tests do not establish real model quality. Also run:

```bash
python main.py --retrieve-only
python main.py
```

For the sample RFP, inspect the evidence and proposal for AES-256,
SOC 2 Type II, MFA, 99.99% uptime, two regions, Salesforce REST
integration and commercial terms.

## Validation performed

The development environment passed dependency consistency checks and
all 11 automated tests.

Real embedding retrieval and real three-agent local inference completed.
One complete run took approximately 4.7 minutes on CPU.

The model invented two chunk labels; the citation audit detected and
marked them. Generated drafts still require human review.

Machine runtime does not measure human review-time savings.

## Troubleshooting

- Connection refused: start Ollama and check `--ollama-url`.
- Missing model: run `ollama pull` for the selected model.
- Blocked downloads: allow the required Hugging Face/Ollama destinations.
  Do not disable TLS verification.
- Read-only home: Ollama requires a writable `~/.ollama` identity directory.
  `OLLAMA_MODELS` changes model storage only.
- Slow inference: increase `--timeout` or use suitable hardware.
- Larger models may improve quality but need more memory.

## Limitations

This is a prototype, not an exhaustive compliance checker.

Top-k retrieval may miss requirements in long RFPs. Similarity scores are
retrieval scores, not probabilities of correctness.

The project has no vendor capability knowledge base. Generated
certification, pricing, SLA and implementation claims remain unverified.

Prompt instructions do not guarantee protection against malicious
document content. Use trusted demonstration documents.

See [the interview guide](docs/INTERVIEW.md) for design explanations,
demo steps and PoC evaluation guidance.
