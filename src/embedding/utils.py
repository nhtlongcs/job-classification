from IPython.utils import io


def get_query_results(pipeline, query: str, top_k: int = 5):
    try:
        with io.capture_output() as _:
            prediction = pipeline.run_query(query, top_k)
        codes = [
            str(doc.meta["code"])
            for doc in prediction["retriever"]["documents"]
        ]
        labels = [
            str(doc.meta["label"])
            for doc in prediction["retriever"]["documents"]
        ]
        return ", ".join(codes), ", ".join(labels)
    except Exception as e:
        print(e)
        return "", ""
