import myModules.tweetreader as tr
import myModules.bayesianhelper as bh
import pandas as pd


# returns true or false depending on whether the model predicted sentiment correctly
def testModelOnTweet(sentimenttext, actualsentiment):
    sentimenttext = tr.separateWords(sentimenttext, None)

    pSentencePositive = bh.pSetGivenSentence( sentimenttext, POS_WORDCOUNT, P_WORDS_POSITIVE, P_POSITIVE)
    pSentenceNegative = bh.pSetGivenSentence( sentimenttext, NEG_WORDCOUNT, P_WORDS_NEGATIVE, P_NEGATIVE)

    sentiment = 1 if pSentencePositive > pSentenceNegative else 0
    return (sentiment == actualsentiment)


def main():

    # TRAINING ------------------------------------------
    data = tr.readCSV('res/train.csv')
    data = tr.sanitizeSentences(data, "sentimentText")

    # get DataFrames filtered by sentiment, then get DataFrames of all words found
    negativewords = tr.separateWords( data[ data['sentiment'] == 0 ], 'sentimentText' )
    positivewords = tr.separateWords( data[ data['sentiment'] == 1 ], 'sentimentText' )
    allwords      = tr.separateWords( data, 'sentimentText' )

    global VOCABULARY    # all words that have occurred as keys, number of occurrences as values
    global POS_WORDCOUNT # all words that have occurred as keys, number of occurrences in positive records as values
    global NEG_WORDCOUNT # all words that have occurred as keys, number of occurrences in negative records as values
    VOCABULARY    = tr.countUniqueWords( allwords , None)
    POS_WORDCOUNT = tr.countUniqueWords( positivewords, VOCABULARY )
    NEG_WORDCOUNT = tr.countUniqueWords( negativewords, VOCABULARY )



    global P_WORDS_POSITIVE # given positive, probabilities of every word in vocabulary occurring
    global P_WORDS_NEGATIVE # given negative, probabilities of every word in vocabulary occurring
    P_WORDS_POSITIVE = bh.pOfWordsGivenSet(POS_WORDCOUNT, VOCABULARY)
    P_WORDS_NEGATIVE = bh.pOfWordsGivenSet(NEG_WORDCOUNT, VOCABULARY)

    global P_POSITIVE #probability of a set being positive
    global P_NEGATIVE #probability of a set being negative
    P_POSITIVE = bh.pOfSet(1, data, 'sentiment')
    P_NEGATIVE = 1 - P_POSITIVE
    


    # TESTING --------------------------------------------
    testingdata = tr.readCSV('res/test.csv')
    testingdata = tr.sanitizeSentences(testingdata, 'sentimentText')

    correct = pd.Series()
    for i in range(len(testingdata)):
        correct.at[i] = (testModelOnTweet( testingdata['sentimentText'][i], testingdata['sentiment'][i]) )

    accuracy = correct.sum() / len(correct)
    print( "accuracy:", accuracy )
    print( "examples of bad classification:")
    print(testingdata[ correct == False ][:10])

    return

main()
