import pandas as pd
from haystack import Document, Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import (
    ChromaEmbeddingRetriever,
)
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder,
    OpenAITextEmbedder,
    OpenAIDocumentEmbedder,
)
from haystack.utils import ComponentDevice, Device, Secret


class RetrieverPipeline:
    def __init__(
        self,
        labels_file: str,
        model: str = "dunzhang/stella_en_1.5B_v5",
        distance_function: str = "cosine",
        instruction: str = "",
    ):
        self.labels_file = labels_file
        self.model = model
        self.distance_function = distance_function
        self.instruction = instruction
        self.document_store = None
        self.query_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        df_labels = self._load_labels(self.labels_file)
        self.document_store = ChromaDocumentStore(
            distance_function=self.distance_function
        )
        device = ComponentDevice.from_single(Device.gpu(id=0))
        # Initialize the document embedder
        document_embedder = SentenceTransformersDocumentEmbedder(
            model=self.model, device=device
        )
        document_embedder.warm_up()

        # Embed and store documents
        documents = [
            Document(
                content=row["description"],
                meta={"code": row["code"], "label": row["label"]},
            )
            for _, row in df_labels.iterrows()
        ]
        documents_with_embeddings = document_embedder.run(documents)[
            "documents"
        ]
        self.document_store.write_documents(documents_with_embeddings)

        # Create and connect components in the pipeline
        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(
                model=self.model, prefix=self.instruction, device=device
            ),
        )
        self.query_pipeline.add_component(
            "retriever",
            ChromaEmbeddingRetriever(document_store=self.document_store),
        )
        self.query_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )

    def _load_labels(self, file_path: str):
        df = pd.read_csv(file_path)
        columns = ["code", "label", "description"]
        return df[columns]

    def run_query(self, query: str, top_k: int = 5):
        return self.query_pipeline.run(
            {"text_embedder": {"text": query}, "retriever": {"top_k": top_k}}
        )


# Example usage:
# pipeline = RetrieverPipeline("public_data/wi_labels.csv")
# results = pipeline.run_query("Your query here")


class OpenAIRetrieverPipeline:
    def __init__(
        self,
        labels_file: str,
        model: str = "text-embedding-3-large",
        distance_function: str = "cosine",
        instruction: str = "",
    ):
        self.labels_file = labels_file
        self.model = model
        self.distance_function = distance_function
        self.instruction = instruction
        self.document_store = None
        self.query_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        df_labels = self._load_labels(self.labels_file)
        self.document_store = ChromaDocumentStore(
            distance_function=self.distance_function
        )
        api_key = Secret.from_token("openai-api-key")
        # Initialize the document embedder
        document_embedder = OpenAIDocumentEmbedder(
            api_key=api_key, model=self.model
        )
        # document_embedder.warm_up()

        # Embed and store documents
        documents = [
            Document(
                content=row["description"],
                meta={"code": row["code"], "label": row["label"]},
            )
            for _, row in df_labels.iterrows()
        ]
        documents_with_embeddings = document_embedder.run(documents)[
            "documents"
        ]
        self.document_store.write_documents(documents_with_embeddings)

        # Create and connect components in the pipeline
        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component(
            "text_embedder",
            OpenAITextEmbedder(api_key=api_key, model=self.model),
        )
        self.query_pipeline.add_component(
            "retriever",
            ChromaEmbeddingRetriever(document_store=self.document_store),
        )
        self.query_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )

    def _load_labels(self, file_path: str):
        df = pd.read_csv(file_path)
        columns = ["code", "label", "description"]
        return df[columns]

    def run_query(self, query: str, top_k: int = 5):
        return self.query_pipeline.run(
            {"text_embedder": {"text": query}, "retriever": {"top_k": top_k}}
        )
