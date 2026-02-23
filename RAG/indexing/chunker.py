CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    content = (text or "").strip()
    if not content:
        return []

    chunks = []
    start = 0
    text_length = len(content)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_text(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    chunks = []

    for page in pages:
        page_chunks = split_text(page.get("text", ""), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in page_chunks:
            chunks.append(
                {
                    "content": chunk,
                    "page": page.get("page", 0),
                    "type": "text",
                    "source": page.get("source"),
                }
            )

    return chunks
