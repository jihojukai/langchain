"""AIZEN finance RAG chain using LangChain for credit risk Q&A."""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate

def build_finance_chain(weaviate_url: str, collection: str) -> RetrievalQA:
    import weaviate as wv
    client = wv.Client(weaviate_url)
    vectorstore = Weaviate(client=client, index_name=collection, text_key="content", embedding=OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )

