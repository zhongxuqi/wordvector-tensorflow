from pymongo import MongoClient
import csv

version = 1

def InitDB():
    client = MongoClient("mongodb://localhost:27017")
    db = client.DeepSearch
    # db.authenticate("zhongxuqi", "1234567")

    wordDict = {}
    wordList = []
    wordReader = csv.reader(open("data/CorpusWordPOSlist.csv", "r"))
    wordReader.__next__()
    for line in wordReader:
        item = {
            "word": line[1],
            "times": 0
        }
        if (db.words.find({"word": item["word"]}).count() == 0) and \
        (item["word"] not in wordDict):
            wordList.append(item)
            wordDict[item["word"]] = item
    db.words.insert(wordList)
    db.words.create_index("word")
    db.probWords.create_index("word")

def ResetDB():
    client = MongoClient("mongodb://localhost:27017")
    client.drop_database("DeepSearch")
    InitDB()
