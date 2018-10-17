__data__ = {}

def sanitizeSentences(sentenceList):
    for sentence in sentenceList:
        print("TODO: sanitize sentences")
        # remove special characters, possibly keeping in emojis if tone can be parsed from them
    return

# takes a list of sentences and breaks them into their individual words, retuned as list
def separateWords(sentenceList):
    for sentence in sentenceList:
        print("TODO: separate words")
        # make pandas dataset of individual words from list of sentences
    return

# take a list of words and return a dict where <key:value> represents <word:count>
def countUniqueWords(list):
    for word in list:
        print("TODO: count unique words")
        # make pandas dataset word/count pairs of individual words
    return

def readCSV(filename):
    __data__ = {}
    with open(filename) as file:
        __data__ = pd.read_csv(file, names=["itemID", "sentiment", "sentimentSrc", "sentimentText"], skiprows=[0], usecols=[0,1,2,3])
        __data__ = __data__.dropna()
        __data__['sentiment'] = __data__['sentiment'].astype("int")

        positives = __data__[ (__data__['sentiment'] == 1) ]
        negatives = __data__[ (__data__['sentiment'] == 0) ]
        print ( positives['sentimentText'])
    return __data__

def getData():
    return __data__