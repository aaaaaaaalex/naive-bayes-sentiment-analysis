import re
import pandas as pd
import numpy as np

# takes a pandas DataFrame and removes any undesireable characters from the column specified, returns a copy of the processed data
def sanitizeSentences(data, columnname):
    copydata = data.copy()
    
    # remove special characters, possibly keeping in emojis if tone can be parsed from them
    copydata.loc[ :, columnname] = copydata.loc[ :, columnname].apply(
        lambda row:
            re.sub( r"[\'\"\\\+\*\!\.\,\?\[\^\]\$\(\)\{\}\=\<\>\|\:\;\&\#\@\`\/\%\-\_\~0-9]", "", str.lower(row) )
    )
    #print( copydata.loc[ :, 'sentimentText'] )
    return copydata


# takes a list of sentences and breaks them into their individual words, retuned a Pandas Frame with column 'words'
def separateWords(data, columnname):
    allwords = np.array([""])
    copydata = data.copy()
    
    for sentence in copydata[columnname]:
        words = np.array( sentence.split(" ") )
        allwords = np.append(allwords, words)

    allwords = allwords[ allwords != "" ]
    return pd.DataFrame(allwords,  columns=['words'])

    
# take a Pandas DataFrame of words and return list where <key:value> represents <word:count>
def countUniqueWords(words):
    uniquewords = pd.value_counts(words) 
    return uniquewords


def readCSV(filename):
    data = {}
    with open(filename) as file:
        data = pd.read_csv(file, names=["itemID", "sentiment", "sentimentSrc", "sentimentText"], skiprows=[0], usecols=[0,1,2,3])
        data = data.dropna()
        data['sentiment'] = data['sentiment'].astype("int")
    return data
