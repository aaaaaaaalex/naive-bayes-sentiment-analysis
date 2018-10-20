import myModules.tweetreader as tr
import myModules.bayesianhelper as bh


# returns true or false depending on whether the model predicted sentiment correctly
def testModelOnTweet(sentimenttext, actualsentiment):
    return


def main():

    # TRAINING ------------------------------------------
    data = tr.readCSV('res/train.csv')
    data = tr.sanitizeSentences(data, "sentimentText")

    negativewords = tr.separateWords( data[ data['sentiment'] == 0 ], 'sentimentText' )
    positivewords = tr.separateWords( data[ data['sentiment'] == 1 ], 'sentimentText' )
    
    vocabulary = tr.countUniqueWords(
        (tr.separateWords( data, 'sentimentText' )).words
        )

    pos_wordcount = tr.countUniqueWords(positivewords.words)
    neg_wordcount = tr.countUniqueWords(negativewords.words)

    # get params for Naive Bayes Theorem: P(positive|word) = (P(word|positive) * P(positive)) / P(word)
    global pWordsPositive = bh.pOfWordsGivenSet(pos_wordcount, vocabulary)
    global pWordsNegative = bh.pOfWordsGivenSet(neg_wordcount, vocabulary)

    global pPositive = bh.pOfSet( 1, data, "sentiment")
    global pNegative = 1 - pPositive
    global pWords = bh.pWords( [pos_wordcount, neg_wordcount] )

    # TESTING --------------------------------------------
    testingdata = tr.readCSV('res/test.csv')
    testingdata = tr.sanitizeSentences(testingdata, 'sentimentText')


    pSentencePositive = bh.pSetGivenSentence(sentence, pWordsPositive, pPositive, pWords)
    pSentenceNegative = bh.pSetGivenSentence(sentence, pWordsNegative, pNegative, pWords)

    print(pSentencePositive, ":", pSentenceNegative)

    return

main()
