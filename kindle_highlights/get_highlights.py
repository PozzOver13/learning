
import pathlib

from kindle_highlights.highlight_parser import Highlight, Book

if __name__ == "__main__":

    current_directory = pathlib.Path.cwd()
    parsed_books = list(
        set(file.stem for file in current_directory.glob("**/*.md"))
    )
    highlight_separator = "=========="
    highlight_json = dict()
    library = []


    with open("My Clippings.txt", "r", encoding='utf-8') as file:
        data = file.read()

    highlights = data.split(highlight_separator)

    for raw_string in highlights:
        h = Highlight(raw_string)
        if h.title not in Book.book_list:
            b = Book(h.title, h.author, h.date)
            b.add_highlight(h.content)
            library.append(b)
        else:
            for b in library:
                if b.title == h.title:
                    b.add_highlight(h.content)

    for book in library:
        if book.title:
            if book.title.strip() not in parsed_books:
                book.write_book(format="markdown")
            else:
                print(f"{book.title} is already written.")

    print("DONE!")