from langchain_google_community import VertexAISearchRetriever

# os.environ["PROJECT_ID"] = "nexapro-439420"
# os.environ["DATA_STORE_ID"] = "eurlex_1745811150807"

retriever = VertexAISearchRetriever(
    # search_engine_id="eurlex-enterprise_1745817682023",
    max_documents=3,
)

# retriever.invoke("What laws should a SaaS startup follow in Germany?")
