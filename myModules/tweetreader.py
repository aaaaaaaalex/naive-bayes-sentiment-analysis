import re
import pandas as pd

# takes a pandas DataFrame and removes any undesireable characters from the column specified, returns a copy of the processed data
def sanitizeSentences(data, columnname):
    copydata = data.copy()
    
    # remove special characters, possibly keeping in emojis if tone can be parsed from them
    copydata.loc[ :, columnname] = copydata.loc[ :, columnname].apply(
        lambda row: 
            re.sub( r"[\'\"\\\+\*\?\[\^\]\$\(\)\{\}\=\<\>\|\:\;\&\#\@\`\/\%\-()]", "", row )
    )

    #print( copydata.loc[ :, 'sentimentText'] )
    return copydata

# takes a list of sentences and breaks them into their individual words, retuned as list
def separateWords(data, columnname):
    

    return

# take a list of words and return a dict where <key:value> represents <word:count>
def countUniqueWords(list):
    for word in list:
        print("TODO: count unique words")
        # make pandas dataset word/count pairs of individual words
    return

def readCSV(filename):
    data = {}
    with open(filename) as file:
        data = pd.read_csv(file, names=["itemID", "sentiment", "sentimentSrc", "sentimentText"], skiprows=[0], usecols=[0,1,2,3])
        data = data.dropna()
        data['sentiment'] = data['sentiment'].astype("int")
    return data
