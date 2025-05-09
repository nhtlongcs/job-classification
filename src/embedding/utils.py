from IPython.utils import io
import warnings
import time

def get_query_results(pipeline, query: str, top_k: int = 5):
    try:
        start = time.time()
        with io.capture_output() as _:
            prediction = pipeline.run_query(query, top_k)
        stop = time.time()
        
        codes = [
            str(doc.meta["code"])
            for doc in prediction["retriever"]["documents"]
        ]
        labels = [
            str(doc.meta["label"])
            for doc in prediction["retriever"]["documents"]
        ]
        return ", ".join(codes), ", ".join(labels), stop - start
    except Exception as e:
        print(e)
        # warnings.warn(f"Error in query: {query}")

        return "", "", 0