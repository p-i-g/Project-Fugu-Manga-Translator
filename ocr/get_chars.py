import manga109api
if __name__ == "__main__":
    manga109_root = "../datasets/Manga109/Manga109_released_2021_12_30"
    dataset = manga109api.Parser(manga109_root)
    chars = set()
    for book in dataset.books:
        for page in dataset.get_annotation(book)["page"]:
            for text in page["text"]:
                for char in text["#text"]:
                    chars.add(char)
    for char in list("1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        chars.add(char)
    print("".join(chars))