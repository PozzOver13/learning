from PyPDF2 import PdfReader, PdfWriter, PdfFileReader, PdfFileWriter

PATH_PDF_INPUT = "Quest Furto.pdf"
PATH_PDF_OUTPUT = "Quest Furto UPDATE.pdf"


def extract_pages(input_pdf_path, output_pdf_path, pages_to_extract):
    # Open the input PDF file
    with open(input_pdf_path, "rb") as input_pdf_file:
        reader = PdfReader(input_pdf_file)
        writer = PdfWriter()

        # Loop through the specified pages and add them to the writer
        for page_num in pages_to_extract:
            if page_num < len(reader.pages):
                page = reader.pages[page_num]
                writer.add_page(page)
            else:
                print(f"Page number {page_num} is out of range. The document has {len(reader.pages)} pages.")

        # Write the output PDF file
        with open(output_pdf_path, "wb") as output_pdf_file:
            writer.write(output_pdf_file)
        print(f"Extracted pages saved to {output_pdf_path}")

if __name__ == "__main__":
    pages_to_extract = [4, 5, 6]  # List of pages to extract (0-indexed)

    extract_pages(PATH_PDF_INPUT, PATH_PDF_OUTPUT, pages_to_extract)