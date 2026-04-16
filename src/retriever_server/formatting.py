def format_retrieval_passages(retrieval_result: list[dict]) -> str:
    formatted = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        body = "\n".join(content.split("\n")[1:])
        formatted += f"Doc {idx + 1}(Title: {title}) {body}\n"
    return formatted
