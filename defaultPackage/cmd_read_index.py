import sys
import timeit

from nltk import SnowballStemmer

start = timeit.default_timer()
stemmer = SnowballStemmer("english")

if len(sys.argv) == 3:
    paramType = sys.argv[1]

    if paramType == '--term':

        termName = stemmer.stem(sys.argv[2])
        termID = -1

        with open("termids.txt", "r", encoding="utf8") as file:
            for line in file:
                line = line.strip().split("\t")
                if len(line) > 0:
                    if line[1] == termName:
                        termID = line[0]
                        break

        if termID == -1:
            print("Term not found!")
        else:
            with open("term_index.txt", "r", encoding="utf8") as file:
                for line in file:
                    if len(line) > 0 and line.split()[0] == termID:
                        print("Listing for term: " + sys.argv[2])
                        print("TERM_ID: " + str(termID))
                        print("Number of documents containing term: " + line.split()[2])
                        print("Term frequency in corpus: " + line.split()[1])

        end = timeit.default_timer()
        print("\nTime : " + str(end - start) + ' seconds')

