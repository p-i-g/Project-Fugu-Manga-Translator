import cv2
import easyocr


def main():
    img = cv2.imread("../datasets/Manga109/Manga109_released_2021_12_30/images/AisazuNihaIrarenai/005.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(["ja"])
    result = reader.readtext(img)
    print(reader.readtext(img, detail=0))


if __name__ == "__main__":
    main()
