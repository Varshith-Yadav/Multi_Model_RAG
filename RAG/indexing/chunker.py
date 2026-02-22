from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

def chunk_text(pages):
    chunks = []
    for p in pages:
        splits = splitter.split_text(p['text'])
        for s in splits:
            chunks.append({
                "content": s,
                "page": p['page'],
                'type': 'text',
                'source': p['source']
            })
    return chunks

