from PyPDF2 import PdfReader, PdfWriter

PATH_PDF_INPUT = "Quest Furto.pdf"
PATH_PDF_OUTPUT = "Quest Furto UPDATE.pdf"

if __name__ == "__main__":
    reader = PdfReader(PATH_PDF_INPUT)
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()


    for id, i in enumerate(text.split("\n")):
        print(id, i)

    modified_text = '\n'.join(lines)
    page.mergePage(PyPDF2.PdfFileReader(
        PyPDF2.PdfFileWriter().add_blank_page(
            width=page.mediaBox.getWidth(),
            height=page.mediaBox.getHeight()
        ).getPage(0)
    ).add_page(page))
    page.mergeTextFrame(PyPDF2.PdfTextWriter().begin_text_frame().write_text(modified_text).end_text_frame())

    # write PDF output
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(PATH_PDF_OUTPUT)