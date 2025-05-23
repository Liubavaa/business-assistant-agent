import os
from typing import Annotated, Optional, Literal, Callable
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain.tools.retriever import create_retriever_tool

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_vertexai import ChatVertexAI
from tavily import TavilyClient

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

os.environ["TAVILY_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "586937ba4cca312a704e4626222cbe414e773fcb"

tavily_client = TavilyClient()

### ---------- State Definition ----------

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    market_analysis: Optional[str]
    risks: Optional[str]
    competitors: Optional[str]
    website_analysis: Optional[str]

### ---------- Utility Functions ----------

_printed = set()

def _print_event(event: dict, _printed: set, max_length=3000):
    """Prints formatted event data for debugging."""
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def handle_tool_error(state) -> dict:
    """Handles errors from tool calls and formats them as messages."""
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list):
    """Wraps tools with a fallback error handler."""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

### ---------- Assistant Wrapper ----------

class Assistant:
    """Generic assistant runner for tool-enabled prompts."""
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            state = {**state}
            result = self.runnable.invoke(state)
            if not result.tool_calls and not result.content:
                state["messages"].append(("user", "Respond with a real output."))
            else:
                break
        try:
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            return {"messages": [ToolMessage(content=result.content, tool_call_id=tool_call_id)]}
        except AttributeError:
            return {"messages": result}

### ---------- Tool Input Models ----------

class ToLawAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle law questions."""
    question: str = Field(description="Question to answer about legal field.")


class ToMarketResearchAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle company market research."""
    company_name: str = Field(description="The name of the company provided by the user.")
    industry: Optional[str] = Field(description="Field in which company are operating (not required).")
    country: Optional[str] = Field(description="Country in which company are operating (not required).")

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "Google",
                "industry": "IT",
                "country": "US",
            }
        }


class ToWebsiteAssistant(BaseModel):
    """Transfer work to a specialized assistant to analyze website or compare with competitor."""
    company_name: str = Field(description="The name of the company provided by the user.")
    competitor_name: Optional[str] = Field(description="Competitor provided by user (not required if user didn't provide it).")
    company_url: Optional[str] = Field(description="The url of the website (not required if user didn't provide it).")
    competitor_url: Optional[str] = Field(description="Competitor url (not required if user didn't provide it).")
    user_question: Optional[str] = Field(description="Any additional information or requests from the user regarding the website (not required if user didn't provide it).")

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "Grammarly",
                "url": "https://www.grammarly.com",
                "competitor": "Google",
                "company_url": "https://www.google.com",
                "request": "What could you recommend to improve readability?",
            }
        }

### ---------- Prompts and Tools ----------

# Law Assistant Prompt and Tools
law_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a specialized assistant for handling law related questions. "
        "The primary assistant delegates work to you when the user needs help with legal issues. "
        "Use available tools persistently to find relevant answers."
    )),
    ("placeholder", "{messages}"),
])

from langchain.tools.retriever import create_retriever_tool
from app.eurlex_retriever import retriever

retrieve_tool = create_retriever_tool(
    retriever,
    name="vertex_ai_standard_retriever",
    description="Searches uploaded eurlex legal documents for relevant information. Use when user asks about EU law, regulations, or company obligations. Recommended to use before another tool",
)
law_tools = [TavilySearchResults(max_results=2), retrieve_tool]


# Primary Assistant Prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "Your primary role is to manage a conversation between assistants. "
        "You can help with a customer requests for company market research, request for website analysis or ask law related question. "
        "In such case delegate the task to the appropriate specialized assistant by invoking the corresponding tool. "
        "Consider previous messages to ensure if current agent should continue operating or another one should be selected. "
        "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
    )),
    ("placeholder", "{messages}"),
])

primary_tools = [
    ToLawAssistant,
    ToWebsiteAssistant,
    ToMarketResearchAssistant
]

### ---------- Graph Assembly ----------

def get_main_graph(model="gemini-2.0-flash"):
    """
    Initializes the full assistant orchestration graph using LangGraph.
    Routes messages between law, market research, and website analysis agents.
    """
    llm = ChatVertexAI(model=model, temperature=0)

    from app.market_research_graph import get_market_research_graph, MarketResearchState
    from app.website_analyzer_graph import get_web_graph, WebsiteAnalysisState

    graph_market_research = get_market_research_graph(model)
    graph_web = get_web_graph(model)

    # Bind models to their tools/prompts
    law_runnable = law_prompt | llm.bind_tools(law_tools)
    assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_tools)

    builder = StateGraph(State)

    # Law assistant routing
    def entry_node(state: State) -> dict:
        return {
            "messages": [
                ToolMessage(
                    content=f"Reflect on the above conversation between the host assistant and the user."
                    f" Use the provided tools to assist the user.",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ]
        }
    builder.add_node("entry_law_assistant", entry_node)
    builder.add_node("law_assistant", Assistant(law_runnable))
    builder.add_edge("entry_law_assistant", "law_assistant")
    builder.add_node("law_tools", create_tool_node_with_fallback(law_tools))
    def route_law_tools(state: State):
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            return "law_tools"
        return END
    builder.add_edge("law_tools", "law_assistant")
    builder.add_conditional_edges("law_assistant", route_law_tools, ["law_tools", END])

    # Market research entry
    def entry_market_research_assistant(state: State):
        tool_call = state["messages"][-1].tool_calls[0]  # Get last tool call

        # The arguments passed from the primary assistant
        args = tool_call.get("args", {})
        if not args or "company_name" not in args:
            raise ValueError("Market research assistant requires at least company_name in tool call args.")

        final_state = graph_market_research.invoke(MarketResearchState(**args))
        print("Final report:", final_state['report'][:200])
        return {
            "messages": [
                ToolMessage(
                    content=f"Here is the market research report: {final_state['report']}",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ],
            "market_analysis": final_state["market_analysis"],
            "risks": final_state["risks"],
            "competitors": final_state["competitors"],
        }
    builder.add_node("entry_market_research_assistant", entry_market_research_assistant)

    # Website analysis entry
    def entry_website_assistant(state: State):
        tool_call = state["messages"][-1].tool_calls[0]

        args = tool_call.get("args", {})
        if not args.get("competitor_name") and state.get("competitors", ""):
            args["competitor_name"] = state["competitors"][0]

        final_state = graph_web.invoke(WebsiteAnalysisState(**args))

        return {
            "messages": [
                ToolMessage(
                    content=f"Here is the website analysis: {final_state['website_analysis']}",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ],
            "website_analysis": final_state["website_analysis"],
        }
    builder.add_node("entry_website_assistant", entry_website_assistant)

    # Primary assistant and routing logic
    builder.add_node("primary_assistant", Assistant(assistant_runnable))

    def route_primary_assistant(state: State,):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        print(tool_calls)
        if tool_calls:
            if tool_calls[0]["name"] == ToLawAssistant.__name__:
                return "entry_law_assistant"
            elif tool_calls[0]["name"] == ToMarketResearchAssistant.__name__:
                return "entry_market_research_assistant"
            elif tool_calls[0]["name"] == ToWebsiteAssistant.__name__:
                return "entry_website_assistant"
        raise ValueError("Invalid route")


    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        [
            "entry_law_assistant",
            "entry_market_research_assistant",
            "entry_website_assistant",
            END,
        ],
    )

    builder.add_edge(START, "primary_assistant")

    main_graph = builder.compile()
    return main_graph
