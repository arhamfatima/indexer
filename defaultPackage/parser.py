import os
import re
import timeit

import bs4
import nltk
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer

path = "C:\\Users\\luky\\PycharmProjects\\Indexer\\corpus"
stopListPath = "C:\\Users\\luky\\PycharmProjects\\Indexer\\stoplist.txt"
filePathsList = []
fileNamesList = []
termIdsByTerm = {}
termIDCount = 0
stemmer = SnowballStemmer("english")
stopListDict = {}
invertedIndexByTermId = {}


class InvertedIndexLine:
    docPositionsById = {}
    termIDCount = 0
    docFrequency = 0
    termFrequency = 0

    def __init__(self, term_id):
        self.termId = term_id


def add_timing(key, value):
    global sumTimesByCategory
    if key not in sumTimesByCategory:
        sumTimesByCategory[key] = value
    else:
        sumTimesByCategory[key] += value


def filter_tags(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'http:', 'a']:
        return False
    elif isinstance(element, bs4.element.Comment):
        return False
    elif isinstance(element, bs4.element.Doctype):
        return False
    elif re.match(r"[\s\r\n]+", str(element)):
        return False
    elif re.match(
            "^(http://www\.|https://www\.|http://|https://)?[a-z0-9]+([\-.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(/.*)?$",
            str(element)):
        return False
    return True


def apply_regex(body):
    soup = BeautifulSoup(body, 'html.parser')
    # print(soup.original_encoding)
    texts = soup.findAll(text=True)
    visible_texts = filter(filter_tags, texts)
    str = u" ".join(t.strip() for t in visible_texts)
    # print(str)
    regex = re.compile('[^a-zA-Z\d ]')
    return regex.sub(' ', str)
    # str = re.sub("[()-:!'’—?&@{}𔄜[\]|%$◄►#©“”=♦…_·€¢•<>`→\"]", " ", str)
    # str = str.strip()
    # return str


def parse(file_contents):
    return apply_regex(file_contents)


def tokenize(parsed_content):
    return nltk.wordpunct_tokenize(parsed_content)


def process(tokens):
    global termIDCount, stopListDict, termIdsByTerm
    words = []

    for token in tokens:
        word = token.lower()
        if word not in stopListDict:
            newWord = stemmer.stem(word)
            words.append(newWord)
            if newWord not in termIdsByTerm:
                termIdsByTerm[newWord] = termIDCount
                termIDCount += 1

    return words


def read_files():
    global stopListDict, fileNamesList, filePathsList

    # read stop_list
    with open(stopListPath, "r") as file:
        stopWords = file.read()
        stopList = stopWords.split()
        stopListDict = dict.fromkeys(stopList, 1)

    # read all file paths
    for root, dirs, files in os.walk(path):
        for file in files:
            fileNamesList.append(file)
            filePathsList.append(os.path.join(root, file))


def main():
    global termIDCount

    read_files()

    # initialize variables
    startTime = timeit.default_timer()
    startIndex = 0
    endIndex = len(filePathsList)

    # for all files
    for i in range(startIndex, endIndex):

        file = open(filePathsList[i], errors='ignore')

        # parsing
        parsedContent = parse(file.read())

        # tokenizing
        tokens = tokenize(parsedContent)

        # assign termID, remove stop words, apply stemming, and lowercase
        words = process(tokens)

        file.close()

        # make inverted docId
        incremented = False

        for term in words:
            termIDCount = termIdsByTerm[term]
            line = None
            if termIDCount not in invertedIndexByTermId:
                line = InvertedIndexLine(termIDCount)
            if not incremented:
                line.docFrequency += 1
                incremented = True
            line.termFrequency += 1


    # write doc ids
    with open("docids.txt", "w", encoding="utf8") as file:
        docId = 0
        for name in fileNamesList:
            file.write(str(docId) + "\t" + name + "\n")
            docId += 1

    # write term ids
    exceptions = 0
    with open("termids.txt", "w", encoding='utf8') as file:
        for key, value in termIdsByTerm.items():
            try:
                file.write(str(value) + "\t" + key + "\n")
            except UnicodeEncodeError:
                exceptions += 1
                continue

    stop = timeit.default_timer()
    print('\n\nExceptions : ' + str(exceptions))
    print('Time: ' + str(round(stop - startTime, 4)))


if __name__ == "__main__":
    main()
