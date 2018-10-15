import pandas as pd


def sanitizeSentences(sentenceList):
    for sentence in sentenceList:
        # remove special characters, possibly keeping in emojis if tone can be parsed from them
    return

def tokenizeWords(sentenceList):
    for sentence in sentenceList:
        # make pandas dataset of individual words from list of sentences
    return

def countUniqueWords(list):
    for word in list:
        # make pandas dataset word/count pairs of individual words
    return

def readCSV(filename):
    data = {}
    with open(filename) as file:
        data = pd.read_csv(file, names=["itemID", "sentiment", "sentimentSrc", "sentimentText"], skiprows=[0], usecols=[0,1,2,3])
        data = data.dropna()
        data['sentiment'] = data['sentiment'].astype("int")

        positives = data[ (data['sentiment'] == 1) ]
        negatives = data[ (data['sentiment'] == 0) ]
        print ( positives['sentimentText'])
    return data

def _main_():
    testDataset = readCSV('train.csv')
_main_()
