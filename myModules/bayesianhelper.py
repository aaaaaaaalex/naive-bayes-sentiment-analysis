"""
Bayesian functions for calulating probabilities given certain conditions
"""
import pandas as pd

# find probability of word occurring out of all words in a set ( P(word|set) )
# takes Series of elements and the total occurrences of each element
# returns a DataFrame of all elements in set, with their probabilities of occurring
def pOfWordsGivenSet( givenset, vocabulary ):
    totaloccurrences = givenset.sum()
    prob_elements = givenset.apply(
        lambda word_occurences: #with laplace smoothing
            (word_occurences + 1) / (totaloccurrences + len(vocabulary))
    )
    return prob_elements


# find probability of a set occurring in a column, out of all records ( P(set) )
def pOfSet(setvalue, allrecords, columnname):
    setOccurrences = allrecords[ allrecords[columnname] == setvalue ]
    return len(setOccurrences) / len(allrecords)


# find probability of words occurring out of all possible words( P(word) )
# returns a DataFrame of all words from all passed sets, with their probabilities of occurring
def pWords( allsets ):
    totaloccurrences = allsets[0]
    for i in range(1, len(allsets) ):
        totaloccurrences = totaloccurrences.add(allsets[i], fill_value=0)

    totalwords = totaloccurrences.sum()
    probWords = totaloccurrences.apply(
        lambda word_occurences:
            word_occurences / totalwords
    )

    return probWords

# determines the probability of a set given a word, based off P(word|set), P(set), and P(word)
def pSetGivenWord(word, pWordsGivenSet, pSet, pWords):
    try:
        p = (pWordsGivenSet[word] * pSet) / pWords[word]
    except KeyError:
        p = pSet
    return p

# determines the probability that a sentence (list of words, not a long string) belongs to a set based on the average p(set|word) across the sentence
def pSetGivenSentence(sentence, pWordsGivenSet, pSet, pWords):
    sentence = sentence.apply(
        lambda row:
            pSetGivenWord(row.lower(), pWordsGivenSet, pSet, pWords )
    )
    sentence = sentence.dropna()
    p = sentence.sum() / len(sentence)
    return p
