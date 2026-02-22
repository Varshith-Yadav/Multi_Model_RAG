def build_prompt(query, docs):
    context = "\n\n".join([
        f"[{d['type']} | page {d['page']}]\n{d['content']}"
        for d in docs
    ])

    return f"""
    You are a multimodal document analyst.

    Use text, tables and image descriptions to answer.

    Context:
    {context}

    Question:
    {query}

    Answer with page references.
    """
