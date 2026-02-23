from pathlib import Path

import streamlit as st

from RAG.augmentation.prompt_builder import build_prompt
from RAG.generation.llm import generate
from RAG.retrieval.retriever import IndexNotReadyError, index_exists, load_index, retrieve
from scripts.ingest import ingest_all

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"


def save_uploaded_pdfs(uploaded_files):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        target_path = RAW_DIR / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(str(target_path))

    return saved_files


def format_source_label(index: int, source_doc: dict) -> str:
    source_type = source_doc.get("type", "unknown")
    page = source_doc.get("page", "n/a")
    return f"Source {index}: {source_type} | page {page}"


def main():
    st.set_page_config(page_title="Multi Model RAG", page_icon=":books:", layout="wide")
    st.title("Multi Model RAG")
    st.caption("Upload PDFs, ingest chunks into FAISS, and ask questions.")

    with st.sidebar:
        st.subheader("System Status")
        st.write(f"Index ready: {'Yes' if index_exists() else 'No'}")

        if st.button("Reload Index"):
            try:
                load_index(force_reload=True)
                st.success("Index loaded.")
            except Exception as exc:
                st.error(str(exc))

    st.subheader("1) Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Uploaded PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file first.")
            else:
                saved = save_uploaded_pdfs(uploaded_files)
                st.success(f"Saved {len(saved)} PDF file(s) to {RAW_DIR}")

    with col2:
        if st.button("Run Ingestion", use_container_width=True):
            try:
                with st.spinner("Ingesting PDFs and building index..."):
                    docs = ingest_all(RAW_DIR)
                    load_index(force_reload=True)
                st.success(f"Ingestion complete. Indexed {len(docs)} chunks.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("2) Ask Questions")

    with st.form("query_form"):
        question = st.text_area("Question", height=100, placeholder="Ask about your documents...")
        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5, step=1)
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Question cannot be empty.")
            return

        try:
            docs = retrieve(question, k=top_k)
            prompt = build_prompt(question, docs)
            answer = generate(prompt)
        except IndexNotReadyError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            return

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")
        for index, doc in enumerate(docs, start=1):
            with st.expander(format_source_label(index, doc), expanded=False):
                st.write(f"Source file: {doc.get('source', 'n/a')}")
                st.write(doc.get("content", ""))

                image_path = doc.get("image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption=image_path, use_container_width=True)


if __name__ == "__main__":
    main()
