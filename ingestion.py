import PyPDF2
from loguru import logger


def read_pdf_file(file_path: str):
    # for now each page as chunck

    with open(file_path, 'rb') as file:
        pdf_data = []
        print(pdf_data)
        pdf_reader = PyPDF2.PdfReader(file)
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text = page.extract_text()
            pdf_data.append(text)

        logger.info(f"read file at {file_path} with char length {
            len(pdf_data)}.")
        return pdf_data


def main():

    read_pdf_file("data/profilicity.pdf")


if __name__ == "__main__":
    main()
