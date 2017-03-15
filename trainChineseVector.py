from pymongo import MongoClient
from WordVector import *
import Util

mContentVectorTrainer = ChineseVectorTrainer()

# read all vocabulary
client = MongoClient("localhost", 27017)
db = client.DeepSearch
mContentVectorTrainer.FeedWords(db.words.find({}))
mContentVectorTrainer.FeedWords(db.probWords.find({}))

lines = []
datafile = open("wiki_cn_out_00", "r")
for i in range(100000):
    lines.append(datafile.readline().replace(" \n","").split(" "))
mContentVectorTrainer.FeedContent(lines)
# print(mContentVectorTrainer.data)
# mContentVectorTrainer.ShowData(1, 8, 2, 1)
mContentVectorTrainer.InitConfig()
mContentVectorTrainer.TrainModal()

# print(mContentVectorTrainer.weights[1, :])
# print(mContentVectorTrainer.biases.shape)

wordList = mContentVectorTrainer.GetResultWordList()
# for index in range(1):
#     print(wordList[index])

for wordItem in wordList:
    Util.UpdateWord(wordItem)
