import pyspark
import re


def pairWords(words: list, key) -> list:
    pairList = list()
    wordsCopy = words.copy()
    wordsCopy.remove(key)
    for word in wordsCopy:
        pairList.append((key, word))
    return pairList


def getWordPairs(words: list) -> list:
    wordsDict = {}
    for word in words:
        if word not in wordsDict.keys():
            wordsDict[word] = pairWords(words, word)
    return list(wordsDict.values())


sc = pyspark.SparkContext('local[*]')
regex = re.compile("[^a-zA-Z\s]")

# Grab file and convert each line into a string in a list
# Convert each of string within the list to all lowercase
# regex makes everything lowercase, removes non alpha characters, and removes whitespace
# Make each list of line strings into a list of lists where the inner list
# is of all the words in the lines
# Finally time for word counts
# Collects the word pairs as permutations
# Each permutation is first added with a 1, then the reduceByKey combines them
# for a total count.
# Uses the number value to perform the sort. That's what the x[1] is for
sortedPairsRDD = sc.textFile('file:////home/josh/Downloads/TestInput.txt')\
    .map(lambda x: x.lower())\
    .map(lambda x: regex.sub("", x))\
    .map(lambda x: x.split())\
    .flatMap(lambda x: getWordPairs(x)).flatMap(lambda x: x)\
    .map(lambda x: (x, 1))\
    .reduceByKey(lambda x, y: x + y)\
    .sortByKey()

# Print it pretty
print("Word pairs in descending order")
count = 1
for pair in sortedPairsRDD.collect():
    print("Pair", count, ":", pair[0], "Count:", pair[1])
    count += 1
