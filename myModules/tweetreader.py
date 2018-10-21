import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

def filterChars(chars):
    chars = chars.lower()
    chars = re.sub( r"[\'\"\\\+\*\.\,\!\?\[\^\]\{\}\=\<\>\|\;\:\(\)\&\#\@\$\`\/\%\-\_\~0-9]", "", chars )
    chars = re.sub( r"[ {2,}]", " ", chars)
    return chars

# takes a pandas DataFrame and removes any undesireable characters from the column specified, returns a copy of the processed data
def sanitizeSentences(data, columnname):
    copydata = data.copy()
    
    # remove special characters, possibly keeping in emojis if tone can be parsed from them
    copydata.loc[ :, columnname] = copydata.loc[ :, columnname].apply(
        lambda row:
            filterChars(row)
    )
    #print( copydata.loc[ :, 'sentimentText'] )
    return copydata[ copydata[columnname] != "" ]


# takes a list of sentences and breaks them into their individual words, returns a Pandas Series with <key:value> pair <index:word>
# if columnname is none, 'data' is assumed to be a single sentence
def separateWords(data, columnname):
    allwords = pd.Series([""]) 

    if columnname is not None:
        copydata = data.copy()
        for sentence in copydata[columnname]:
            words = pd.Series( sentence.split(" ") )
            allwords = allwords.append(words, ignore_index=True)
    else:
        words = pd.Series( data.split(" "))
        allwords = allwords.append( words , ignore_index=True)

    allwords = allwords[ allwords != "" ]
    return allwords

    
# converts a Pandas.Series of all word occurrences into a wordcount of every unique word
# if 'vocabulary' is provided, any vocabulary words not found in 'words' will be added to the output Series with count:0
def countUniqueWords(words, vocabulary):
    uniquewords = pd.value_counts(words)
    if vocabulary is None:
        return uniquewords
    
    # the below operations have the effect of merging the two series - 0 is added to any words that are already counted, non-present values are added as 0
    zeroedvocabulary = vocabulary.apply(
        lambda wordcount:
            wordcount*0
    )
    uniquewords = uniquewords.add( zeroedvocabulary , fill_value=0)
    return uniquewords.sort_values(ascending = False)


def readCSV(filename):
    data = {}
    with open(filename) as file:
        data = pd.read_csv(file, names=["itemID", "sentiment", "sentimentSrc", "sentimentText"], skiprows=[0], usecols=[0,1,2,3])
        data = data.dropna()
        data['sentiment'] = data['sentiment'].astype("int")
    return data
