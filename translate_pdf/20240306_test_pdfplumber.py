import pdfplumber

if __name__ == "__main__":
    with pdfplumber.open("Quest Furto.pdf") as pdf:
        first_page = pdf.pages[0]
        # list_lines = first_page.extract_text_lines()
        dict_lines = first_page.extract_text().split("\n")

        rows_cleaned = []
        for page in pdf.pages:
            # extract tables in page 1
            tables_page_n = page.extract_tables()

            for id_table, table in enumerate(tables_page_n):
                print(f"table extracting: {id_table}")

                for row in table:
                    # exclude missing values
                    row_temp = [x for x in row if x != '' and x != ' ' and x != None]
                    # remove unwanted characters
                    row_temp = [x.replace('\n', ' ') for x in row_temp]
                    # remove "\uf063 SI" and "\uf063 NO"
                    row_temp = [x.replace('\uf063 SI', 'SI') for x in row_temp]
                    row_temp = [x.replace('\uf063 NO', 'NO') for x in row_temp]
                    # remove yes and no
                    row_temp = [x for x in row_temp if x != 'SI']
                    row_temp = [x for x in row_temp if x != 'NO']

                    rows_cleaned.append(row_temp)

            # flatten the list
        rows_without_missing_flatten = [item for sublist in rows_cleaned for item in sublist]

        print("questions extracted: ")
        for i in rows_without_missing_flatten:
            print(f"{i}| ")








