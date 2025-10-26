import PyPDF2
from loguru import logger
import chromadb
from chromadb.utils import embedding_functions


colbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="colbert-ir/colbertv2.0")


def read_pdf_file(file_path: str) -> list[str]:

    with open(file_path, 'rb') as file:
        pdf_data = []
        pdf_reader = PyPDF2.PdfReader(file)
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text = page.extract_text()
            pdf_data.append(text)

        logger.info(f"read file at {file_path} with char length {
            len(pdf_data)}.")
        # logger.info(pdf_data[15])
        return pdf_data


def create_chroma_db(chunks: list[str]):
    chroma_client = chromadb.PersistentClient("data/chroma_db")

    collection = chroma_client.create_collection(
        name="rag_collection",
        embedding_function=colbert_ef
    )

    documents = read_pdf_file("data/profilicity.pdf")
    collection.add(documents=documents, ids=[
                   str(i) for i in range(len(documents))])


def get_chroma_db(path: str):
    client = chromadb.PersistentClient("data/chroma_db")
    collection = client.get_collection(
        name="rag_collection",
        embedding_function=colbert_ef)
    return collection


def main():


if __name__ == "__main__":
    main()
