import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Initializing RAG System...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
documents = SimpleDirectoryReader(input_files=["sample_rfp.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
@tool("Search RFP Document")
def search_rfp(query: str) -> str:
    """
    Use this tool to search the client's RFP document for specific requirements.
    Input should be a specific question like 'What are the security requirements?'
    """
    response = query_engine.query(query)
    return str(response)

print("Initializing Agent Workforce...")
agent_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# 1. Define the Specialized Agents
security_agent = Agent(
    role="Information Security Architect",
    goal="Identify all security requirements in the RFP and draft a compliant response.",
    backstory="You are a strict security expert. You must search the RFP for encryption, compliance (like SOC2), and access rules, then write a response proving our system meets these standards.",
    verbose=True,
    llm=agent_llm,
    tools=[search_rfp] # Hands the RAG tool to the agent
)

technical_agent = Agent(
    role="Cloud Systems Engineer",
    goal="Identify all technical and uptime requirements in the RFP and draft a system architecture response.",
    backstory="You are a senior engineer. Search the RFP for uptime, failover, and API integration requirements, and explain how our cloud platform handles them.",
    verbose=True,
    llm=agent_llm,
    tools=[search_rfp]
)

proposal_editor = Agent(
    role="Lead Proposal Manager",
    goal="Combine the drafted sections into a professional, cohesive final proposal.",
    backstory="You are the final reviewer. You take the highly technical drafts from your team and format them into a clean, persuasive business proposal ready to send to the client.",
    verbose=True,
    llm=agent_llm
)

# 2. Define the Tasks
task_security = Task(
    description="Search the RFP for security requirements and draft the 'Security & Compliance' section of our proposal.",
    expected_output="A 2-paragraph draft explaining how we meet their encryption, MFA, and audit requirements.",
    agent=security_agent
)

task_technical = Task(
    description="Search the RFP for technical requirements and draft the 'System Architecture & Uptime' section of our proposal.",
    expected_output="A 2-paragraph draft explaining how we meet their uptime guarantees, failover regions, and API needs.",
    agent=technical_agent
)

task_compile = Task(
    description="Review the drafts from the Security and Technical teams. Combine them into a final proposal document with an introduction and conclusion.",
    expected_output="A complete, professional Markdown document ready for the client.",
    agent=proposal_editor,
    context=[task_security, task_technical], # Forces the editor to wait for the drafts
    output_file="Final_RFP_Response.md"
)

rfp_crew = Crew(
    agents=[security_agent, technical_agent, proposal_editor],
    tasks=[task_security, task_technical, task_compile],
    verbose=True
)

if __name__ == "__main__":
    print("--- Starting Automated RFP Response ---")
    rfp_crew.kickoff()
    print("--- Complete! Check Final_RFP_Response.md ---")