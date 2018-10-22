# A collection of functions that all serve the purpose of manipulating sentences
# with the aim of making Naive Bayesian classification more effective

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer

# removes all special characters from input
def filterSpecialChars(sentence):
    sentence = sentence.lower()
    sentence = re.sub( r"&(.{1,5});", "" ,sentence) #HTML-Entities
    sentence = re.sub( r"[\'\"\\\+\*\.\,\!\?\[\^\]\{\}\=\<\>\|\;\:\(\)\&\#\@\$\`\/\%\-\_\~0-9]", "", sentence )
    return sentence

# removes spaces made of more than one whitespace char, replaces them with a single whitespace
def filterMultipleSpaces(sentence):
    sentence = re.sub( r" +"   , " ", sentence )
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

            #internally call the spaces filter to ensure output isnt mangled
            sentence = filterMultipleSpaces(sentence)
            return sentence
        except LookupError:
            print("--------Downloading stopwords library from NLTK...")
            nltk.download('stopwords')
            print("--------Done! Continuing execution...")

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
    negations = [
            {'regex': r"[\s(^)]not\s"  , 'replacewith': " not-"},
            {'regex': r"[\s(^)]no\s"   , 'replacewith': " no-"},
            {'regex': r"[\s(^)]didnt\s", 'replacewith': " didnt-"},
            {'regex': r"[\s(^)]wont\s" , 'replacewith': " wont-"},
        ]
    for n in negations:
        sentence = re.sub( n['regex'], n['replacewith'], sentence )
    return sentence


# pass words in a sentence through a lemmatizer and return the processed sentence
def lemmatizeWords(sentence):
    while True:
        try:
            lemmatizer = WordNetLemmatizer()
            words = pd.Series(data=sentence.split(" "))
            words = words.apply(
                    lambda word:
                        lemmatizer.lemmatize(word)
                    )

            sentence = " ".join(words.values)
            return sentence
        except LookupError:
            print("--------Downloading wordnet from NLTK...")
            nltk.download('wordnet')
            print("--------Done!")




