import myModules.tweetreader as tr

def main():
    data = tr.readCSV('res/train.csv')
    data = tr.sanitizeSentences(data, "sentimentText")
    allwords = tr.separateWords(data, 'sentimentText')
    #wordcount = tr.countUniqueWords(wordcount)
    
    
    #print(allwords[0:100])

    return


main()
