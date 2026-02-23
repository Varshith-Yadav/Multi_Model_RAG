from pathlib import Path

import streamlit as st

from RAG.augmentation.prompt_builder import build_prompt
from RAG.generation.llm import generate
from RAG.retrieval.retriever import IndexNotReadyError, index_exists, load_index, retrieve
from scripts.ingest import ingest_all

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5


def save_uploaded_pdfs(uploaded_files):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        target_path = RAW_DIR / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(str(target_path))

    return saved_files


def render_sources(sources):
    if not sources:
        return

    with st.expander("Sources"):
        for index, source in enumerate(sources, start=1):
            source_type = source.get("type", "unknown")
            page = source.get("page", "n/a")
            file_path = source.get("source", "n/a")

            st.markdown(f"**{index}. {source_type} | page {page}**")
            st.caption(file_path)

            snippet = source.get("content", "")
            if snippet:
                st.write(snippet[:600] + ("..." if len(snippet) > 600 else ""))

            image_path = source.get("image_path")
            if image_path and Path(image_path).exists():
                st.image(image_path, caption=image_path, use_container_width=True)

            st.divider()


def run_query(question: str, top_k: int):
    docs = retrieve(question, k=top_k)
    prompt = build_prompt(question, docs)
    answer = generate(prompt)
    return answer, docs


def main():
    st.set_page_config(page_title="Multi Model RAG Chat", page_icon=":speech_balloon:", layout="wide")
    init_state()

    st.title("Multi Model RAG Chat")
    st.caption("Upload PDFs, ingest, then chat with your documents.")

    with st.sidebar:
        st.subheader("System")
        st.write(f"Index ready: {'Yes' if index_exists() else 'No'}")
        st.session_state.top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=st.session_state.top_k)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reload Index", use_container_width=True):
                try:
                    load_index(force_reload=True)
                    st.success("Index loaded.")
                except Exception as exc:
                    st.error(str(exc))
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.subheader("Data")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Save PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one PDF first.")
            else:
                saved = save_uploaded_pdfs(uploaded_files)
                st.success(f"Saved {len(saved)} file(s).")

        if st.button("Run Ingestion", use_container_width=True):
            try:
                with st.spinner("Ingesting documents..."):
                    ingest_all(RAW_DIR)
                    load_index(force_reload=True)
                st.success("Ingestion completed.")
            except Exception as exc:
                st.error(str(exc))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, sources = run_query(question, st.session_state.top_k)
            st.markdown(answer)
            render_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
        except IndexNotReadyError as exc:
            error_message = str(exc)
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_message}", "sources": []})
        except Exception as exc:
            error_message = f"Query failed: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})


if __name__ == "__main__":
    main()
