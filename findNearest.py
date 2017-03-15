import numpy as np
from pymongo import MongoClient

client = MongoClient("localhost", 27017)
db = client.DeepSearch

wordDict = {}
for wordObject in db.words.find({}):
    wordDict[wordObject["word"]] = wordObject
for wordObject in db.probWords.find({}):
    wordDict[wordObject["word"]] = wordObject

testList = ["一", "二","三","游戏","北京","上海","上海市","杭州"]
for testWord in testList:
    if testWord in wordDict:
        testWordObject = wordDict[testWord]
        if "vector" not in testWordObject:
            continue
        print(testWord + ":")
        testWordVector = np.array(testWordObject["vector"], np.float32)
        nearList = []
        for word in wordDict:
            if testWord == word:
                continue
            wordObject = wordDict[word]
            if "vector" not in wordObject:
                continue
            wordVector = np.array(wordObject["vector"], np.float32)
            value = np.linalg.norm(testWordVector-wordVector)
            wordObject["value"] = value
            hasInserted = False
            index = 0
            for item in nearList:
                if item["value"] > wordObject["value"]:
                    hasInserted = True
                    break
                index += 1
            if hasInserted:
                nearList.insert(index, wordObject)
                if len(nearList) > 10:
                    nearList.pop()
            elif len(nearList) < 10:
                nearList.append(wordObject)
        wordOutputList = []
        for item in nearList:
            wordOutputList.append(item["word"])
        print(wordOutputList)

measureList = [("一", "二"),("一","怎么")]
for pairItem in measureList:
    if (pairItem[0] in wordDict) and (pairItem[1] in wordDict):
        item1 = wordDict[pairItem[0]]
        item2 = wordDict[pairItem[1]]
        if ("vector" in item1) and ("vector" in item2):
            print(pairItem,":")
            wordVector1 = np.array(item1["vector"], np.float32)
            wordVector2 = np.array(item2["vector"], np.float32)
            print(np.linalg.norm(wordVector1-wordVector2))

ListLen = 100
vectorNormList = []
for word in wordDict:
    wordObject = wordDict[word]
    if "vector" not in wordObject:
        continue
    index = 0
    for item in vectorNormList:
        if np.linalg.norm(wordObject["vector"]) < np.linalg.norm(item["vector"]):
            break
        index += 1
    if index < ListLen:
        vectorNormList.insert(index, wordObject)
        if len(vectorNormList) > ListLen:
            vectorNormList.pop()
print([item["word"] for item in vectorNormList])

wordTimesList = []
for word in wordDict:
    wordObject = wordDict[word]
    if "times" not in wordObject:
        continue
    index = 0
    for item in wordTimesList:
        if wordObject["times"] > item["times"]:
            break
        index += 1
    if index < ListLen:
        wordTimesList.insert(index, wordObject)
        if len(wordTimesList) > ListLen:
            wordTimesList.pop()
print([item["word"] for item in wordTimesList])
