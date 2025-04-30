import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import GoogleSerperAPIWrapper

import urllib.request
from bs4 import BeautifulSoup
import re

os.environ["SERPER_API_KEY"] = "586937ba4cca312a704e4626222cbe414e773fcb"
serper_search = GoogleSerperAPIWrapper()


class CompanyUrl(BaseModel):
    company_url: str = Field(description="The url of the company website.")


class WebsiteAnalysisState(BaseModel):
    company_name: str = Field(description="The name of the company provided by the user.")
    company_url: Optional[str] = Field(default=None, description="The url of the company website.")
    company_content: Optional[str] = Field(default=None, description="The content of the company website.")

    competitor_name: Optional[str] = Field(default=None, description="The competitor name of the company website.")
    competitor_url: Optional[str] = Field(default=None, description="The competitor url of the company website.")
    competitor_content: Optional[str] = Field(default=None, description="The content of the competitor website.")

    user_question: Optional[str] = Field(default=None,
                                         description="Any additional information or requests from the user regarding the website.")
    website_analysis: Optional[str] = Field(default=None, description="The analysis result.")


def search_company_url(name: str, llm) -> str:
    """Search for official website of a company or competitor by name."""
    search_results = serper_search.results(query=f"{name} official website", n_results=3)
    if not search_results:
        raise ValueError(f"No search results found for {name}.")

    result = llm.with_structured_output(CompanyUrl).invoke(
        f"From these search results, pick the most official domain URL for company {name}: {search_results}"
    ).model_dump()
    print(result)
    return result["company_url"]


def scrape_url(url: str) -> str:
    """Scrape website content given a url."""
    try:
        import urllib
        req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
        con = urllib.request.urlopen(req, timeout=10)

        html = con.read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ")
        text = re.sub("[ \t]+", " ", text)
        text = re.sub("\\s+\n\\s+", "\n", text)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def search_user_company_url(state: WebsiteAnalysisState, llm):
    print(f"Searching for {state.company_name} company website.")
    return {"company_url": search_company_url(state.company_name, llm)}


def search_competitor_url(state: WebsiteAnalysisState, llm):
    print(f"Searching for {state.competitor_name} company website.")
    return {"competitor_url": search_company_url(state.competitor_name, llm)}


def scrape_company(state: WebsiteAnalysisState):
    print(f"Searching for {state.company_name} company website.")
    return {"company_content": scrape_url(state.company_url)}


def scrape_competitor(state: WebsiteAnalysisState):
    print(f"Searching for {state.competitor_name} company website.")
    return {"competitor_content": scrape_url(state.competitor_url)}


def analyze_or_compare(state: WebsiteAnalysisState, llm):
    print(f"Analyzing {state.company_name, state.competitor_name} company website.")

    system_prompt = f"""
You are a professional website analyst.

Company Website Content:
{state.company_content or 'Not available'}

{f"Competitor Website Content:\n{state.competitor_content}" if state.competitor_content else ''}

{"Please perform a general analysis" if not state.competitor_content else 'Please perform a comparison'}
{f" considering user request: {state.user_question}." if state.user_question else '.'}

Provide a detailed, clear output.
"""
    result = llm.invoke(system_prompt)
    return {"website_analysis": result.content}


def get_web_graph(model: str = "gemini-2.0-flash") -> CompiledStateGraph:
    import os
    os.environ["OPENAI_API_KEY"] = "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"
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    builder_web = StateGraph(WebsiteAnalysisState)

    builder_web.add_node("search_company_url", lambda state: search_user_company_url(state, llm))
    builder_web.add_node("search_competitor_url", lambda state: search_competitor_url(state, llm))
    builder_web.add_node("scrape_company", scrape_company)
    builder_web.add_node("scrape_competitor", scrape_competitor)
    builder_web.add_node("analyze", lambda state: analyze_or_compare(state, llm))

    def route_web_research(state: WebsiteAnalysisState):
        next_nodes = []
        if not state.company_url:
            next_nodes.append("search_company_url")
        else:
            next_nodes.append("scrape_company")

        if state.competitor_name:
            if not state.competitor_url:
                next_nodes.append("search_competitor_url")
            else:
                next_nodes.append("scrape_competitor")

        print("Next nodes:", next_nodes)
        return next_nodes

    gathering_data_nodes = ["search_company_url", "scrape_company", "search_competitor_url", "scrape_competitor"]
    builder_web.add_conditional_edges(
        START,
        route_web_research,
        gathering_data_nodes
    )

    builder_web.add_edge("search_company_url", "scrape_company")
    builder_web.add_edge("search_competitor_url", "scrape_competitor")

    builder_web.add_edge("scrape_company", "analyze")
    builder_web.add_edge("scrape_competitor", "analyze")

    builder_web.add_edge("analyze", END)

    graph_web = builder_web.compile()
    return graph_web
