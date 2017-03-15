from pymongo import MongoClient
from WordVector import *
from langPreprocesser.ChinesePreprocesser import *

mContentVectorMachine = ChineseVectorMachineV2()

# read all vocabulary
client = MongoClient("localhost", 27017)
db = client.DeepSearch
mContentVectorMachine.FeedWords(db.words.find({}), all_accept=True)
mContentVectorMachine.FeedWords(db.probWords.find({}))

# init vector machine
mContentVectorMachine.InitConfig(skip_window=2, block_steps=-1, iscontinue=False)
mContentVectorMachine.StartDrawing()
# load data
mLangReader = ChineseReader()
fileList = ["wiki_cn_format_00", "wiki_cn_format_01", "wiki_cn_format_03"]
for item in fileList:
    datafile = open(item, "r")
    while True:
        num_article = 0
        for line in datafile:
            mLangReader.FeedContent(line)
            if line == "</end>\n":
                num_article += 1
            if num_article >= 1000:
                break
        if len(mLangReader.ArticleList) == 0:
            break
        mContentVectorMachine.ArticleList.extend(mLangReader.ArticleList)
        mLangReader.Clean()
        mContentVectorMachine.StartRunning(isTest=False)

datafile.close()

mContentVectorMachine.SaveState()
