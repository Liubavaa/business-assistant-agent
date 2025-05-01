Multi-Agent Business Desicion-Making Assistant

## Overview

The assistant supports three main functionalities:

- 🔍 **Legal Document Retrieval and Question Answering**  
  A legal agent retrieves and answers questions based on EU legal documents (Eur-Lex).

- 📊 **Market Research and Competitor Analysis**  
  An agent gathers missing company information, performs market analysis, identifies risks, and lists competitors using real-time search.

- 🌐 **Website Scraping and Comparison**  
  Another agent analyzes or compares company websites (and optionally competitors) based on scraped content.

## Repository Structure

> ⚠️ **Note:** This project was developed using the [agent-starter-pack](https://github.com/langchain-ai/agent-starter-pack), which provides a production-ready agent deployment scaffold. Most infrastructure and utility code are inherited from that starter pack.

### My Contributions

All custom logic and agent workflows are implemented in the `app/` directory:

| File | Description |
|------|-------------|
| `app/main_graph.py` | Main entrypoint that coordinates the different agent workflows using LangGraph. |
| `app/market_research_graph.py` | Market research agent that uses web search to enrich company data and analyze industry context. |
| `app/website_analyzer_graph.py` | Web analyzer agent that scrapes and compares websites of companies and competitors. |
| `app/eurlex_retriever.py` | Custom retriever for querying EU legal documents via the Eur-Lex corpus. |

> 🛠 `server.py` was part of the original starter pack and has been slightly modified to integrate with this multi-agent system.

### Evaluation

A dedicated evaluation notebook is included:

- `notebooks/evaluating_langgraph_agent.ipynb`  
  Contains sample queries and showcases how the agent performs across real-world tasks.

## How to Use

Please refer to the built-in README provided in the `business-agent/` directory (from `agent-starter-pack`) for setup instructions, environment configuration, and deployment guidelines.

---
