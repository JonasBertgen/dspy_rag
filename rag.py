import dspy
from loguru import logger
from ingestion import get_chroma_db

# todo move to ingestion.py


class ChromaRetriever(dspy.Retrieve):
    def __init__(self, collection, k=3):
        super().__init__(k=k)
        self.collection = collection

    def forward(self, query: str) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=self.k)
        logger.info(results['documents'])
        return results['documents']


class RAG(dspy.Signature):
    """Answer questions based on retrieved context."""
    context = dspy.InputField(desc="Retrieved context passages")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question")

# Define RAG module


class SimpleRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generator = dspy.Predict(RAG)

    def forward(self, question):
        context = self.retriever(question)
        response = self.generator(context=context, question=question)
        return response


lm = dspy.LM("ollama_chat/mistral",
             api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)


collection = get_chroma_db("data/chroma_db")
retriever = ChromaRetriever(collection, k=3)
rag = SimpleRAG(retriever)

result = rag("Is Donald Trump authentic?")
print(result["answer"].strip())
