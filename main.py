import myModules.tweetreader as tr
import myModules.bayesianhelper as bh
import pandas as pd


# returns true or false depending on whether the model predicted sentiment correctly
def testModelOnTweet(sentimenttext, actualsentiment):
    sentimenttext = tr.separateWords(sentimenttext, None)

    try:
        pSentencePositive = bh.pSetGivenSentence( sentimenttext, P_WORDS_POSITIVE, P_POSITIVE, P_WORDS )
        pSentenceNegative = bh.pSetGivenSentence( sentimenttext, P_WORDS_NEGATIVE, P_NEGATIVE, P_WORDS )
        sentiment = 1 if pSentencePositive > pSentenceNegative else 0
        return (sentiment == actualsentiment)

    except ZeroDivisionError:
        print("--------EXCEPTION-----\n",sentimenttext.words)
        print("LENGTH:", len(sentimenttext.words))



def main():

    # TRAINING ------------------------------------------
    data = tr.readCSV('res/train.csv')
    data = tr.sanitizeSentences(data, "sentimentText")

    # get DataFrames filtered by sentiment, then get DataFrames of all words found
    negativewords = tr.separateWords( data[ data['sentiment'] == 0 ], 'sentimentText' )
    positivewords = tr.separateWords( data[ data['sentiment'] == 1 ], 'sentimentText' )
    allwords      = tr.separateWords( data, 'sentimentText' )

    # get Pandas Series's where each word in the vocabulary is a key, and the corresponding value is word count
    # including the full vocabulary as an argument will add any unpresent characters to the wordcount with count 0
    vocabulary    = tr.countUniqueWords( allwords , None)
    pos_wordcount = tr.countUniqueWords( positivewords, vocabulary )
    neg_wordcount = tr.countUniqueWords( negativewords, vocabulary )



    # get params for Naive Bayes Theorem: P(positive|word) = ( P(word|positive) * P(positive) ) / P(word)
    #                                     P(negative|word) = ( P(word|negative) * P(negative) ) / P(word)
    global P_WORDS_POSITIVE 
    P_WORDS_POSITIVE = bh.pOfWordsGivenSet(pos_wordcount, vocabulary)
    return
    global P_WORDS_NEGATIVE
    P_WORDS_NEGATIVE = bh.pOfWordsGivenSet(neg_wordcount, vocabulary)

    global P_POSITIVE
    P_POSITIVE = bh.pOfSet( 1, data, "sentiment")
    global P_NEGATIVE
    P_NEGATIVE = 1 - P_POSITIVE
    global P_WORDS
    P_WORDS = bh.pWords( [pos_wordcount, neg_wordcount] )

    # TESTING --------------------------------------------
    testingdata = tr.readCSV('res/test.csv')
    testingdata = tr.sanitizeSentences(testingdata, 'sentimentText')

    correct = pd.Series()
    for i in range(len(testingdata)):
        correct.at[i] = (testModelOnTweet( testingdata['sentimentText'][i], testingdata['sentiment'][i]) )

    accuracy = correct.sum() / len(correct)
    print( "accuracy:", accuracy )
    print( "examples of correct classification:")
    print(testingdata[ correct ][:10])

    return

main()
