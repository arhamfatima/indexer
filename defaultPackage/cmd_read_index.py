import sys
import timeit

from nltk import SnowballStemmer

start = timeit.default_timer()
stemmer = SnowballStemmer("english")

while True:
    #  ./read_index.py --term apple
    if len(sys.argv) == 3:
        paramType = sys.argv[1]

        if paramType == '--term':

            termName = stemmer.stem(sys.argv[2])
            termIDCount = -1

            with open("termids.txt", "r", encoding="utf8") as file:
                for line in file:
                    line = line.strip().split("\t")
                    if len(line) > 0:
                        if line[1] == termName:
                            termIDCount = line[0]
                            break

            if termIDCount == -1:
                print("Term not found!")
            else:
                with open("term_index.txt", "r", encoding="utf8") as file:
                    for line in file:
                        if len(line) > 0 and line[0] == termIDCount:
                            print("Listing for term: " + sys.argv[2])
                            print("TERM_ID: " + str(termIDCount))
                            print("Number of documents containing term: " + line[1])
                            print("Term frequency in corpus: " + line[2])
                            break

            end = timeit.default_timer()
            print("\nTime : " + str(end - start))

    elif len(sys.argv) == 1:
        paramType = sys.argv[1]
        if paramType == 'exit':
            break
