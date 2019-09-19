import os
import re
import sys
import timeit
from io import StringIO

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

sumTimes = {}


def add_timing(key, value):
    global sumTimes
    if key not in sumTimes:
        sumTimes[key] = value
    else:
        sumTimes[key] += value


class InvertedIndexLine:
    termID = 0
    termFrequency = 0
    lastWrittenDocId = 0

    def __init__(self, term_id):
        self.termId = term_id
        self.docPositionsById = {}


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

    sumSpeeds = 0

    # for all files
    for docId in range(startIndex, endIndex):

        loopStartTime = timeit.default_timer()

        file = open(filePathsList[docId], errors='ignore')

        # parsing
        time = timeit.default_timer()
        parsedContent = parse(file.read())
        add_timing('Parsing', timeit.default_timer() - time)

        # tokenizing
        time = timeit.default_timer()
        tokens = tokenize(parsedContent)
        add_timing('Tokenizing', timeit.default_timer() - time)

        # assign termID, remove stop words, apply stemming, and lowercase
        time = timeit.default_timer()
        words = process(tokens)
        add_timing('Stem&Lower', timeit.default_timer() - time)

        file.close()

        # make inverted index
        time = timeit.default_timer()
        uniqueWords = {}
        for term in words:

            # skip this word if already written
            if term in uniqueWords:
                continue

            # add to dictionary
            uniqueWords[term] = 1

            termID = termIdsByTerm[term]

            # if not in global dictionary
            if termID not in invertedIndexByTermId:
                termLine = InvertedIndexLine(termID)
            else:
                termLine = invertedIndexByTermId[termID]

            # generate positions
            positions = [str(i + 1) for i, x in enumerate(words) if x == term]

            termLine.termFrequency += len(positions)

            termLine.docPositionsById[docId] = positions

            invertedIndexByTermId[termID] = termLine

        add_timing('MakingInvertedIndex', timeit.default_timer() - time)

        # calculate speed
        currSpeed = timeit.default_timer() - loopStartTime
        sumSpeeds += currSpeed
        if docId % 5 == 0:
            print('\rElapsed Time : ' + str(round(timeit.default_timer() - startTime, 1)), end='')
            print(' Progress : ' + str(docId) + '/' + str(endIndex - 1), end='')
            print(' Remaining Time : ' + str(round((endIndex - 1 - docId) * (sumSpeeds / (docId + 1)), 0)), end='')
            sys.stdout.flush()

    time = timeit.default_timer()

    fileStr = StringIO()

    # write inverted index
    # for each term in dictionary
    for termID, termLine in invertedIndexByTermId.items():

        # write properties of this term
        fileStr.write(str(termID) + '\t' + str(termLine.termFrequency) + '\t' + str(len(termLine.docPositionsById)) + '\t')
        firstDocId = -1

        # for each doc containing this term
        for docID, positions in termLine.docPositionsById.items():

            deltaEncodedDocId = docID
            if firstDocId != -1:
                deltaEncodedDocId = docID - firstDocId
            else:
                firstDocId = docID

            firstPosition = True

            # for each position
            for position in positions:
                if firstPosition:
                    fileStr.write(str(deltaEncodedDocId) + ',' + str(position) + '\t')
                    firstPosition = False
                else:
                    fileStr.write('0,' + str(position) + '\t')

        fileStr.write('\n')

    add_timing('WritingIndexToRam', timeit.default_timer() - time)

    time = timeit.default_timer()

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

    with open("term_index.txt", "w", encoding='utf8') as file:
        file.write(fileStr.getvalue())

    add_timing('WritingToFiles', timeit.default_timer() - time)

    stop = timeit.default_timer()
    print('\n\nExceptions : ' + str(exceptions))
    print('Time: ' + str(round(stop - startTime, 4)))

    for key in sumTimes:
        print(key + ' : ' + str(round(sumTimes[key], 3)))


if __name__ == "__main__":
    main()
