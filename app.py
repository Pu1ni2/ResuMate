import os
from typing import TypedDict, List

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, END


# ----------------------------
# 1) LangGraph State
# ----------------------------
class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# ----------------------------
# 2) Vectorstore / Retriever
# ----------------------------
def get_vectordb():
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "about_me")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def format_docs(docs: List[Document]) -> str:
    # Keep it simple + readable. Include sources for better grounded answers.
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        locator = f"{src}" + (f", page {page}" if page is not None else "")
        blocks.append(f"[{i}] ({locator})\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


# ----------------------------
# 3) Graph Nodes
# ----------------------------
def retrieve_node_factory(vectordb: Chroma, k: int = 6):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    def retrieve(state: RAGState):
        q = state["question"]
       # docs = retriever.get_relevant_documents(q)
        docs = retriever.invoke(q)
        return {"context": format_docs(docs)}

    return RunnableLambda(retrieve)


def build_generate_chain():
    system_prompt = """
You are a professional recruiter / personal assistant representing the candidate.
Answer questions about the candidate using ONLY the provided context (resume + personal bio).
Be confident, concise, and professional.

Rules:
- If the answer is not in the context, say you don't have that information.
- Do not invent employers, dates, schools, or metrics.
- When helpful, summarize in bullet points.
- If asked for contact info, only provide what appears in the context.
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer as the candidate's representative:"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    # This returns an AIMessage; LangGraph can stream messages from it
    return prompt | llm


def generate_node(state: RAGState, chain):
    # chain.invoke returns an AIMessage
    msg = chain.invoke({"question": state["question"], "context": state["context"]})
    return {"answer": msg.content}


def build_graph(vectordb: Chroma):
    retrieve_runnable = retrieve_node_factory(vectordb=vectordb, k=6)
    generate_chain = build_generate_chain()

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_runnable)

    # Wrap generate so we can still stream from the underlying LLM chain
    graph.add_node("generate", RunnableLambda(lambda s: generate_node(s, generate_chain)))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ----------------------------
# 4) Streamlit UI
# ----------------------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Set it in your environment or a .env file.")
        st.stop()

    st.set_page_config(page_title="Ask About Me (RAG)", page_icon="🧠", layout="centered")
    st.title("🧠 Ask About Me (RAG)")
    st.caption("Answers are grounded in your resume.pdf + details.txt via ChromaDB + LangGraph.")

    # Ensure DB exists
    chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
    if not os.path.exists(chroma_dir):
        st.warning("Chroma DB not found. Run `python ingest.py` first.")
        st.stop()

    # Cache heavy objects
    @st.cache_resource
    def _load_graph():
        vectordb = get_vectordb()
        return build_graph(vectordb)

    graph = _load_graph()

    question = st.text_input("Ask a question about the candidate:", placeholder="e.g., What are my strongest skills for an AI Engineer role?")
    go = st.button("Ask")

    if go and question.strip():
        placeholder = st.empty()
        streamed_text = ""

        # We stream graph events. For best token streaming, use stream_mode="messages".
        # We accumulate assistant tokens as they arrive.
        initial_state: RAGState = {"question": question.strip(), "context": "", "answer": ""}

        try:
            for event in graph.stream(initial_state, stream_mode="messages"):
                # event is typically (message, metadata)
                msg, meta = event
                # We only show assistant content deltas
                if getattr(msg, "content", None):
                    streamed_text += msg.content
                    placeholder.markdown(streamed_text)

            # Final answer will also be present in the end-state, but we already displayed it.
        except Exception as e:
            st.error(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
