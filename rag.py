import dspy
import chromadb
from chromadb.utils import embedding_functions
from ingestion import read_pdf_file

# todo move to ingestion.py
chroma_client = chromadb.Client()


colbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="colbert-ir/colbertv2.0")

collection = chroma_client.create_collection(
    name="rag_collection",
    embedding_function=colbert_ef
)

documents = ["Python is a high-level programming language known for its simplicity and readability.",
             "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
             "DSPy is a framework for programming with foundation models using declarative modules.",
             "ChromaDB is an open-source embedding database designed for AI applications.",
             "ColBERT is a neural retrieval model that uses contextualized late interaction over BERT."]

collection.add(documents=documents, ids=[str(i)
               for i in range(len(documents))])


class ChromaRetriever(dspy.Retrieve):
    def __init__(self, collection, k=3):
        super().__init__(k=k)
        self.collection = collection

    def forward(self, query: str) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=self.k)
        return results['documents'][0]


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
        self.generator = dspy.ChainOfThought(RAG)

    def forward(self, question):
        # Retrieve relevant documents
        context = self.retriever(question)
        # Generate answer using retrieved context
        response = self.generator(context=context, question=question)
        return response


lm = dspy.LM("ollama_chat/mistral",
             api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

retriever = ChromaRetriever(collection, k=2)
rag = SimpleRAG(retriever)

result = rag("what is dspy?")
print(result)
