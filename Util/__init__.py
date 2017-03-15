from pymongo import MongoClient
import re
import threading
import time

client = MongoClient("mongodb://localhost:27017")
db = client.DeepSearch
wordsDict = None
probWordsDict = None
mutex = threading.Lock()

def reconnect():
    global client, db
    client = MongoClient("mongodb://localhost:27017")
    db = client.DeepSearch

def initDict():
    global wordsDict, probWordsDict
    wordsDict = {}
    for wordItem in db.words.find({}):
        wordsDict[wordItem["word"]] = wordItem
    probWordsDict = {}
    for wordItem in db.probWords.find({}):
        probWordsDict[wordItem["word"]] = wordItem

def checkInit():
    if (wordsDict == None) or (probWordsDict == None):
        initDict()

checkInit()

def GetWordInfo(word):
    return wordsDict[word]

def HasWordInfo(word):
    return word in wordsDict

def HasWordInfoWithUpdate(word):
    global mutex
    if mutex.acquire() and HasWordInfo(word):
        wordObject = GetWordInfo(word)
        db.words.update({
            "word": word,
        },{
            "$set": {"times": wordObject["times"] + 1}
        })
        wordObject["times"] += 1
        mutex.release()


# def HasWordInfoButNHF(word):
#     if db.words.find_one({"word": word, "attr": {"$nin": ["nhf"]}}):
#         return True
#     else:
#         return False

def GetProbWord(word):
    return probWordsDict[word]

def HasProbWordInfo(word):
    return word in probWordsDict

def HasFregProbWordInfo(word):
    if (word in probWordsDict) and probWordsDict[word]["times"] > 9:
        return True
    else:
        return False

def IsWord(word):
    if (word in wordsDict) or (word in probWordsDict):
        return True
    else:
        return False

def IsActualWord(word):
    if (word in wordsDict) or ((word in probWordsDict) and probWordsDict[word]["times"] > 9):
        return True
    else:
        return False

def UpdateWordFreg(word, cnt):
    global mutex
    if mutex.acquire():
        if HasWordInfo(word):
            wordObject = GetWordInfo(word)
            db.words.update({
                "word": word,
            },{
                "$set": {"times": wordObject["times"] + cnt}
            })
            wordObject["times"] += cnt
        elif HasProbWordInfo(word):
            wordObject = GetProbWord(word)
            db.probWords.update({
                "word": word,
            },{
                "$set": {"times": wordObject["times"] + cnt}
            })
            wordObject["times"] += cnt
        else:
            wordObject = {
                "word": word,
                "times": cnt,
            }
            db.probWords.insert(wordObject)
            probWordsDict[word] = wordObject
        mutex.release()

def UpdateWord(WordObject):
    global mutex
    if mutex.acquire():
        if HasWordInfo(WordObject["word"]):
            db.words.update({
                "word": WordObject["word"],
            }, {
                "$set": WordObject
            })
            wordsDict[WordObject["word"]] = WordObject
        elif HasProbWordInfo(WordObject["word"]):
            db.probWords.update({
                "word": WordObject["word"],
            }, {
                "$set": WordObject
            })
            probWordsDict[WordObject["word"]] = WordObject
        mutex.release()

def StoreMaybeWord(word):
    global mutex
    if mutex.acquire():
        if HasProbWordInfo(word):
            probWord = GetProbWord(word)
            db.probWords.update({
                "word": word,
            },{
                "$set": {"times": probWord["times"] + 1}
            })
            probWord["times"] += 1
        else:
            wordObject = {
                "word": word,
                "times": 1
            }
            db.probWords.insert(wordObject)
            probWordsDict[wordObject["word"]] = wordObject
        mutex.release()

def HasChineseWord(word):
    if len(re.findall("[\u4e00-\u9fa5]+",word)) > 0:
        return True
    else:
        return False

def HasWord(word):
    if len(re.findall("[\u4e00-\u9fa5a-zA-Z]+",word)) > 0:
        return True
    else:
        return False

def IsValidForWiki(word):
    if HasChineseWord(word) and (word.find("<doc") < 0):
        return True
    else:
        return False
