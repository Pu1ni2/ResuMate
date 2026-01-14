import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from ingest import load_documents, chunk_documents, build_chroma



if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it in Streamlit → Settings → Secrets.")
    st.stop()


chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
collection_name = os.getenv("CHROMA_COLLECTION", "about_me")

if not os.path.exists(chroma_dir):
    st.info("Chroma DB not found. Building it now from resume.pdf and details.txt...")

    with st.spinner("Ingesting documents and building vector DB..."):
        resume_path = os.getenv("RESUME_PATH", "resume.pdf")
        bio_path = os.getenv("BIO_PATH", "details.txt")

        docs = load_documents(resume_path, bio_path)
        chunks = chunk_documents(docs)

        build_chroma(
            chunks=chunks,
            persist_dir=chroma_dir,
            collection_name=collection_name,
            reset=False,
        )

    st.success("Chroma DB built successfully. Reloading app...")
    st.rerun()


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
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        locator = f"{src}" + (f", page {page}" if page is not None else "")
        blocks.append(f"[{i}] ({locator})\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)



def build_rag_components(k: int = 6):
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    system_prompt = """
You are a professional recruiter / personal assistant representing the candidate.
Answer questions about the candidate using ONLY the provided context (resume + personal bio).
Be confident, concise, and professional. Answer in detailed ,refer both resume and bio.

Rules:
- If the answer is not in the context, say you don't have that information.
- Do not invent employers, dates, schools, or metrics.
- When helpful, summarize in bullet points.
- If asked for contact info, only provide what appears in the context.
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}\n\nAnswer as the candidate's representative:",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    return retriever, prompt, llm


def retrieve_context(retriever, question: str) -> str:
    
    docs = retriever.invoke(question)
    return format_docs(docs)



def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Set it in your environment or a .env file.")
        st.stop()

    st.set_page_config(page_title="ResuMate", page_icon="🧠", layout="centered")
    st.title("🧠ResuMate")
    st.caption("")

    chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
    if not os.path.exists(chroma_dir):
        st.warning("Chroma DB not found. Run `python ingest.py` first.")
        st.stop()

 
    @st.cache_resource
    def _load_components():
        return build_rag_components(k=6)

    retriever, prompt, llm = _load_components()

    question = st.text_input(
        "Ask a question about Punith:",
        placeholder="e.g., What are my strongest skills for an AI Engineer role?",
    )
    go = st.button("Ask")

    if go and question.strip():
        
        with st.spinner("Retrieving relevant context..."):
            context = retrieve_context(retriever, question.strip())

        # with st.expander("Retrieved Context"):
        #     st.text(context)

        placeholder = st.empty()
        streamed_text = ""

        try:
            # Stream tokens from LLM
            for chunk in llm.stream(prompt.format_messages(question=question.strip(), context=context)):
                if getattr(chunk, "content", None):
                    streamed_text += chunk.content
                    placeholder.markdown(streamed_text)

        except Exception as e:
            st.error(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
