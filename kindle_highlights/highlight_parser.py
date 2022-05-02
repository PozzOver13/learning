import os
import re
from unidecode import unidecode

MONTH_MAP = {'gennaio': '01',
             'febbraio': '02',
             'marzo': '03',
             'aprile': '04',
             'maggio': '05',
             'giugno': '06',
             'luglio': '07',
             'agosto': '08',
             'settembre': '09',
             'ottobre': '10',
             'novembre': '11',
             'dicembre': '12'}

class Book:
    book_list = set()

    def __init__(self, title, author, date):
        self.author = author
        self.title = title
        self.date = date
        self.highlights = []
        Book.book_list.add(self.title)

    def add_highlight(self, highlight):
        if highlight:
            self.highlights.append(highlight)

    def __str__(self):
        return f"<Book Object>Title:{self.title}\tAuthor:{self.author}\tHighlights:{len(self.highlights)}"

    def write_book(self, format="markdown"):
        if self.title == None or len(self.highlights) == 0:
            print(f"Not writting because name is None.")
            return False
        clean_title = "".join(
            [c for c in self.title if c.isalpha() or c.isdigit() or c == " "]
        ).rstrip()
        month = MONTH_MAP[self.date.split()[4].lower()]
        year = self.date.split()[5]

        if not os.path.exists(f"books_{year}"):
            os.makedirs(f"books_{year}")

        with open(f"books_{year}/{month}_{clean_title}.md", "w+") as file:
            file.write(f"# {clean_title}")
            file.write("\n")
            for h in self.highlights:
                clean_text = h.replace("\n", " ")
                file.write(f"- {unidecode(clean_text, 'utf-8')}")
                file.write("\n")

            file.close()


class Highlight:
    total_highlights = 0

    def __init__(self, raw_string):
        (
            self.title,
            self.author,
            self.content,
            self.date
        ) = Highlight.parse_single_highlight(raw_string)

    def __str__(self):
        return f"<Highlight Object> Title:{self.title}\tAuthor:{self.author}\tContent:{self.content}"

    @staticmethod
    def parse_single_highlight(highlight_string):
        splitted_string = list(filter(None, highlight_string.split("\n")))

        if len(splitted_string) != 3:
            return None, None, None, None

        # first parse
        author_line = splitted_string[0]
        content = splitted_string[-1]
        info_line = splitted_string[-2]

        # parse author and title
        regex = r"\((.*?)\)"
        match = re.search("\((.*)\)", author_line)

        if not match:
            return None, None, None, None

        author = match.group(1)
        title = author_line[: match.start()]
        date = info_line.split("|")[-1][10:]

        return title, author, content, date


