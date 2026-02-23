import fitz


def load_pdf(path: str):
    pages = []

    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            pages.append(
                {
                    "page": i,
                    "text": page.get_text(),
                    "source": path,
                }
            )

    return pages

