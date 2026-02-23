def extract_tables(pdf_path: str):
    try:
        import camelot
    except ImportError as exc:
        raise RuntimeError(
            "camelot-py is not installed. Install dependencies from requirements.txt."
        ) from exc

    tables = camelot.read_pdf(pdf_path, pages="all")
    results = []

    for table in tables:
        page = table.page
        if isinstance(page, str) and page.isdigit():
            page = int(page) - 1

        results.append(
            {
                "page": page,
                "table": table.df,
                "source": pdf_path,
            }
        )

    return results

