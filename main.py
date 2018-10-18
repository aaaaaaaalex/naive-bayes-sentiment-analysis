import myModules.tweetreader as tr

def main():
    data = tr.readCSV('res/train.csv')
    data = tr.sanitizeSentences(data, "sentimentText")
    worddictionary = tr.separateWords(data, 'sentimentText')
    
    return


main()
