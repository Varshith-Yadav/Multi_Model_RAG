import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RAG.indexing.build_index import build_index
from RAG.indexing.chunker import chunk_text
from RAG.indexing.image_extractor import extract_images
from RAG.indexing.pdf_loader import load_pdf
from RAG.indexing.table_extractor import extract_tables
from RAG.multimodel.image_captioner import caption_image
from RAG.multimodel.table_parser import table_to_text

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"


def process_pdf(pdf_path: str):
    """Process a single PDF into multimodal chunks."""
    all_docs = []
 
    print(f"\nProcessing: {pdf_path}")

    pages = load_pdf(pdf_path)
    text_chunks = chunk_text(pages)
    print(f"  [OK] Extracted {len(text_chunks)} text chunks")
    all_docs.extend(text_chunks)

    try:
        tables = extract_tables(pdf_path)
        table_docs = []

        for table in tables:
            table_text = table_to_text(table["table"])
            table_docs.append(
                {
                    "content": table_text,
                    "type": "table",
                    "page": table["page"],
                    "source": table.get("source", pdf_path),
                }
            )

        print(f"  [OK] Extracted {len(table_docs)} tables")
        all_docs.extend(table_docs)

    except Exception as exc:
        print(f"  [WARN] Table extraction failed: {exc}")

    try:
        images = extract_images(pdf_path)
        image_docs = []

        for image in tqdm(images, desc="  Captioning images"):
            image_path = image["image_path"]
            caption = caption_image(image_path)

            image_docs.append(
                {
                    "content": caption,
                    "type": "image",
                    "page": image["page"],
                    "source": image.get("source", pdf_path),
                    "image_path": image_path,
                }
            )

        print(f"  [OK] Extracted {len(image_docs)} image captions")
        all_docs.extend(image_docs)

    except Exception as exc:
        print(f"  [WARN] Image extraction failed: {exc}")

    return all_docs


def ingest_all(raw_data_dir=RAW_DATA_DIR):
    """Process all PDFs in data/raw and build vector index."""
    raw_dir = Path(raw_data_dir)
    os.makedirs(raw_dir, exist_ok=True)

    pdf_files = sorted(str(path) for path in raw_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {raw_dir}")

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    all_documents = []
    for pdf_path in pdf_files:
        docs = process_pdf(pdf_path)
        all_documents.extend(docs)

    print(f"\nTotal documents prepared: {len(all_documents)}")
    print("\nBuilding FAISS index...")
    indexed_count = build_index(all_documents)
    print(f"Index built successfully with {indexed_count} documents.")

    return all_documents


if __name__ == "__main__":
    ingest_all()
