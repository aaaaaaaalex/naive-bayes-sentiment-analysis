# A collection of functions that all serve the purpose of manipulating sentences
# with the aim of making Naive Bayesian classification more effective

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords as sw


# removes all special characters from input
def filterSpecialChars(sentence):
    sentence = sentence.lower()
    sentence = re.sub( r"&(.{1,5});", "" ,sentence)
    sentence = re.sub( r"[\'\"\\\+\*\.\,\!\?\[\^\]\{\}\=\<\>\|\;\:\(\)\&\#\@\$\`\/\%\-\_\~0-9]", "", sentence )
    return sentence

# removes spaces made of more than one whitespace char, replaces them with a single whitespace
def filterMultipleSpaces(sentence):
    sentence = re.sub( r" +", " ", sentence)
    return sentence


# removes all stop-words from the input sentence
def filterStopwords(sentence):
    while True:
        try:
            stopwords = pd.Series(index=sw.words('english'))
            inputwords = pd.Series( sentence.split(" ") )
            inputwords = inputwords.apply(
                lambda word:
                    _checkIsStopword(word, stopwords)
            )

            sentence = " ".join(inputwords)
            return sentence
        except LookupError:
            nltk.download('stopwords')
            print("-------done!")

# returns the word unchanged if not a stopword, if it is, returns blank string
def _checkIsStopword(word, stopwords):
    try:
        stopwords[word]
        return ""
    except KeyError:
        return word



def tokenizeEmojis(sentence):
    emojis = {
        'highsentiment' : [r"[\s(^)]\:\)", r"[\s(^)]\:D", r"[\s(^)]xD", r"[\s(^)]\(\:", r"[\s(^)]\:\-\)", r"[\s(^)]\(\-\:", r"[\s(^)]\^\.\^", r"[\s(^)]\^\_\^"],
        'lowsentiment'  : [r"[\s(^)]\:\(", r"[\s(^)]\)\:", r"[\s(^)]D\:", r"[\s(^)]\:\'\(", r"[\s(^)]\)\'\:", r"[\s(^)]\-\_\-", r"[\s(^)]T\_T", r"[\s(^)]\;\_\;"]}
    for e in emojis['highsentiment']:
        sentence = re.sub( e, " highsentiment", sentence)

    for e in emojis['lowsentiment'] :
        sentence = re.sub(e,  " lowsentiment",  sentence )

    return sentence


def createNegationFeatures(sentence):
    print("TODO: negation")
