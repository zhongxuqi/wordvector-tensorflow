from base import *
from SentenceGramMachine import *
import Util
from langPreprocesser.ChinesePreprocesser import *

mMachine = SentenceGramMachineV1()
mMachine.FeedWords(Util.db.words.find({}), all_accept=True)
mMachine.FeedWords(Util.db.probWords.find({}), all_accept=False)
mMachine.InitConfig()
mMachine.DrawGraph()

mLangReader = ChineseReader()
datafile = open("format_file_out", "r")
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
    mMachine.FeedArticles(mLangReader.ArticleList)
    mLangReader.Clean()
    mMachine.TrainModel()
mMachine.SaveModel()
