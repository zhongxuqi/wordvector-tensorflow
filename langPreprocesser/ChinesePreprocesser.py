import Util
import _io
from base import *

class ChinesePreprocesser:
    def __init__(self):
        self.ArticleList = []
        self.ArticleItem = Article()
        self.ArticleList.append(self.ArticleItem)
        self.wordDict = {}

    def FeedRawContent(self, rawContent):
        self.ArticleItem.FeedRawContent(rawContent)

    def NewActicle(self):
        if len(self.ArticleItem.sentences) > 0:
            self.ArticleItem = Article()
            self.ArticleList.append(self.ArticleItem)

    def ShowContent(self):
        for item in self.ArticleItem.sentences:
            print(item.words)

    def Write2File(self, outfile):
        if not isinstance(outfile, _io.TextIOWrapper):
            raise Exception("outfile is not instance of File.")
        for article in self.ArticleList:
            if len(article.sentences) > 0:
                for sentence in article.sentences:
                    for word in sentence.words:
                        if word in self.wordDict:
                            self.wordDict[word] += 1
                        elif Util.HasWord(word):
                            self.wordDict[word] = 1
                    outfile.write(sentence.toSpiltLine())
                    outfile.write("\n")
                outfile.write(Article.ArticleEnd)

    def Clean(self):
        self.wordDict.clear()
        self.ArticleList.clear()

class ChineseReader:
    def __init__(self):
        self.ArticleList = []
        self.ArticleItem = Article()

    def FeedContent(self, content):
        if isinstance(content, str):
            self.FeedSentence(content)
        elif isinstance(content, list):
            self.FeedSentences(content)

    def FeedSentence(self, sentence):
        if sentence == Article.ArticleEnd:
            self.NewActicle()
        else:
            SentenceObject = Sentence()
            SentenceObject.FeedFormatSentence(sentence)
            self.ArticleItem.sentences.append(SentenceObject)

    def FeedSentences(self, sentenceList):
        for sentence in sentenceList:
            self.FeedSentence(sentence)

    def NewActicle(self):
        if len(self.ArticleItem.sentences) > 0:
            self.ArticleList.append(self.ArticleItem)
            self.ArticleItem = Article()

    def NextActicle(self):
        if len(self.ArticleList) > 0:
            return self.ArticleList.pop(0)
        else:
            return None

    def Clean(self):
        self.ArticleList.clear()
