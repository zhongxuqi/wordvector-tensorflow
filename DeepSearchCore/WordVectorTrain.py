from pymongo import MongoClient
from WordVector import *
from langPreprocesser.ChinesePreprocesser import *
import Util
import threading

def doRunningBackGround(input_file, options=None, isTest=False):
    Util.reconnect()

    batch_size = 128
    embedding_size = 256
    train_steps = 128
    skip_window = 1
    num_sampled = 64
    block_steps = -1
    num_block = 1000
    iscontinue = False

    # setup the params
    if options is not None:
        for item in options:
            keyvalue = item.split("=")
            if len(keyvalue) != 2:
                continue
            try:
                if keyvalue[0] == "batch_size":
                    batch_size = int(keyvalue[1])
                elif keyvalue[0] == "embedding_size":
                    embedding_size = int(keyvalue[1])
                elif keyvalue[0] == "train_steps":
                    train_steps = int(keyvalue[1])
                elif keyvalue[0] == "skip_window":
                    skip_window = int(keyvalue[1])
                elif keyvalue[0] == "num_sampled":
                    num_sampled = int(keyvalue[1])
                elif keyvalue[0] == "block_steps":
                    block_steps = int(keyvalue[1])
                elif keyvalue[0] == "num_block":
                    num_block = int(keyvalue[1])
                elif keyvalue[0] == "iscontinue":
                    if str(True) == keyvalue[1]:
                        iscontinue = True
            except ValueError as e:
                print(e)

    mContentVectorMachine = ChineseVectorMachineV2()

    # read all vocabulary
    client = MongoClient("localhost", 27017)
    db = client.DeepSearch
    mContentVectorMachine.FeedWords(db.words.find({}))
    mContentVectorMachine.FeedWords(db.probWords.find({}))

    # init the train machine
    mContentVectorMachine.InitConfig(batch_size, embedding_size, train_steps, \
    skip_window, num_sampled, block_steps, num_block, iscontinue)
    mContentVectorMachine.StartDrawing()
    # load data
    mLangReader = ChineseReader()
    while True:
        article_num = 0
        for line in input_file:
            mLangReader.FeedContent(line)
            if line == Article.ArticleEnd:
                article_num += 1
            if article_num >= 500:
                break
        if len(mLangReader.ArticleList) == 0:
            break
        mContentVectorMachine.ArticleList.extend(mLangReader.ArticleList)
        mLangReader.Clean()
        mContentVectorMachine.StartRunning(isTest)

    wordsList = mContentVectorMachine.GetResultWordList()
    for wordItem in wordsList:
        Util.UpdateWord(wordItem)

def running(input_file, options=None, isTest=True):
    t = threading.Thread(target=doRunningBackGround, args=(input_file, options, isTest))
    t.start()
