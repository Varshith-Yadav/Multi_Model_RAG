import camelot
import pandas as pd

def extract_tables(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all')
    results = []

    for t in tables:
        results.append({
            "page": t.page,
            "table": t.df,
            'source': pdf_path
        })

    return results

