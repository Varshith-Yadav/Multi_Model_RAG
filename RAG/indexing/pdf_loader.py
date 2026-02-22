import fitz

def load_pdf(path):
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        pages.append({
            "page": i,
            "text": page.get_text(),
            'source': path
        })
    return pages

