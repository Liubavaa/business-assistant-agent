import os
from typing import Optional, Any
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain_google_vertexai import ChatVertexAI


os.environ["TAVILY_API_KEY"] = "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"
tavily_client = TavilyClient()

### ---------- State & Schema Definitions ----------

class BasicInfo(BaseModel):
    """Structure which llm need to fill out in the first node"""
    company_name: str = Field(description="The name of the company provided by the user.")
    industry: str = Field(default=None, description="Field in which company are operating.")
    country: str = Field(default=None, description="Country in which company are operating.")
    founding_date: str = Field(default=None, description="Date of company founding.")
    product_description: str = Field(default=None,
                                     description="Description of company main product or service.")


class MarketAnalysis(BaseModel):
    """Structure which llm need to fill out in the last node"""
    market_analysis: Optional[str] = Field(default=None, description="Detailed market analysis for company.")
    risks: Optional[str] = Field(default=None, description="Detailed risks for company in the industry.")
    competitors: Optional[str] = Field(default=None, description="Competitors for company.")


class MarketResearchState(BaseModel):
    """State of graph"""
    company_name: str = Field(description="The name of the company provided by the user.")
    industry: Optional[str] = Field(default=None, description="Field in which company are operating.")
    country: Optional[str] = Field(default=None, description="Country in which company are operating.")
    founding_date: Optional[str] = Field(default=None, description="Date of company founding.")
    product_description: Optional[str] = Field(default=None,
                                               description="Description of company main product or service.")

    search_results: Optional[list[Any]] = Field(default=None, description="Search results")

    market_analysis: Optional[str] = Field(default=None, description="Detailed market analysis for company.")
    risks: Optional[str] = Field(default=None, description="Detailed risks for company in the industry.")
    competitors: Optional[str] = Field(default=None, description="Competitors for company.")

    report: Optional[str] = Field(default=None, description="Report for company.")

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "Google",
                "industry": "IT",
                "country": "US",
                "founding_date": "2019-09-22",
                "product_description": "Cloud provider",
            }
        }

### ---------- Node Functions ----------

def get_basic_company_info(state: MarketResearchState, llm) -> MarketResearchState:
    """Get missing basic company info using web search and update state."""

    # Prepare search queries for missing fields
    search_queries = []
    if not state.industry:
        search_queries.append(f"Basic information for {state.company_name} company")

    if not search_queries:
        return state  # Nothing to search for

    search_results = [
        tavily_client.search(search_query, max_results=2, include_raw_content=False, topic="general")["results"]
        for search_query in search_queries]

    system_prompt = f"""
Fill out the following fields if they are missing from the provided search results.

Required fields:
- Company name: {state.company_name if state.company_name else "missing"}
- Industry: {state.industry if state.industry else "missing"}
- Country: {state.country if state.country else "missing"}
- Founding Date: {state.founding_date if state.founding_date else "missing"}
- Product Description: {state.product_description if state.product_description else "missing"}

Only fill in what you find. Keep the output structured.

Sources:
{search_results}
"""
    extracted_info = llm.with_structured_output(schema=BasicInfo).invoke(system_prompt)

    # Update state only with new values
    for field_name, value in extracted_info.model_dump().items():
        if not getattr(state, field_name) and value:
            setattr(state, field_name, value)

    return state



def search_market_info(state: MarketResearchState) -> dict[str, Any]:
    """Search for market analysis, risks, and competitors related to the company."""

    search_queries = [
        f"{state.industry} industry market analysis in {state.country} 2025",
        f"{state.industry} industry risks in {state.country} 2025",
        f"competitors in {state.industry} industry in {state.country} 2025",
    ]

    search_results = [tavily_client.search(query, max_results=3, include_raw_content=True, topic="general")["results"]
                      for query in search_queries]

    return {
        "search_results": search_results
    }


def analyze_market_info(state: MarketResearchState, llm) -> dict[str, Any]:
    """Analyze search results and fill market_analysis, risks, competitors."""

    system_prompt = f"""
Using the provided search results, extract:

- A brief up-to-date Market Analysis (5-6 sentences)
- Key Risks faced by {state.company_name} (bullet points)
- Relevant Main Competitors

Sources:
{state.search_results}
"""

    extracted_info = llm.with_structured_output(schema=MarketAnalysis).invoke(system_prompt).model_dump()

    return {
        "market_analysis": extracted_info["market_analysis"],
        "risks": extracted_info["risks"],
        "competitors": extracted_info["competitors"],
    }


def generate_research_report(state: MarketResearchState) -> dict[str, Any]:
    """Generate a professional market research report based on gathered data."""

    report = f"""
# Company Market Research Report: {state.company_name}

## Basic Information
- **Industry**: {state.industry or "Unknown"}
- **Country**: {state.country or "Unknown"}
- **Founding Date**: {state.founding_date or "Unknown"}
- **Product Description**: {state.product_description or "Unknown"}

## Market Analysis
{state.market_analysis or "Not available."}

## Risks
{state.risks or "Not available."}

## Competitors
{state.competitors or "Not available."}

---

*Report generated automatically by the assistant.*
"""
    return {"report": report}

### ---------- Graph Construction ----------

def get_market_research_graph(model: str = "gemini-2.0-flash") -> CompiledStateGraph:
    """
    Creates a LangGraph pipeline for market research:
    1. Fills missing company info
    2. Searches industry data
    3. Analyzes info
    4. Generates a research report
    """
    llm = ChatVertexAI(model=model, temperature=0)

    builder_onboard = StateGraph(MarketResearchState)
    builder_onboard.add_node("get_basic_company_info", lambda state: get_basic_company_info(state, llm))
    builder_onboard.add_node("search_market_info", search_market_info)
    builder_onboard.add_node("analyze_market_info", lambda state: analyze_market_info(state, llm))
    builder_onboard.add_node("generate_research_report", generate_research_report)

    builder_onboard.add_edge(START, "get_basic_company_info")
    builder_onboard.add_edge("get_basic_company_info", "search_market_info")
    builder_onboard.add_edge("search_market_info", "analyze_market_info")
    builder_onboard.add_edge("analyze_market_info", "generate_research_report")
    builder_onboard.add_edge("generate_research_report", END)

    graph_market_research = builder_onboard.compile()
    return graph_market_research
