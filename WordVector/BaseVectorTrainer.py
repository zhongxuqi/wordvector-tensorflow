import Util
from langPreprocesser.ChinesePreprocesser import Article, Sentence
from six.moves import xrange
from base import *

class BaseVectorTrainer(BaseVocabularyMachine):
    def __init__(self):
        BaseVocabularyMachine.__init__(self)
        self.data = []

    def FeedContent(self, content):
        for line in content:
            for wordItem in line:
                self.data.append(wordItem)

    def GetResultWordList(self):
        wordList = []
        for index in xrange(self.length):
            wordItem = self.reverse_dictionary[index + 1]
            wordItem["weight"] = self.weights[index, :].tolist()
            wordItem["biases"] = self.biases[index].tolist()
            del wordItem["_id"]
            wordList.append(wordItem)
        return wordList

class BaseVectorMachine(BaseVocabularyMachine):
    def __init__(self):
        BaseVocabularyMachine.__init__(self)
        self.ArticleList = []
        self.CurrArtWordList = []
        self.CurrArticle = None
        self.vectors = None

    def FeedArticle(self, article):
        if not isinstance(article, Article):
            raise Exception("article is not Article.")
        if len(article.sentences) == 0:
            raise Exception("article is empty.")
        self.ArticleList.append(article)

    def FeedArticles(self, articles):
        for article in articles:
            self.FeedArticle(article)

    def _nextArticle(self):
        if len(self.ArticleList):
            self.CurrArtWordList = []
            self.CurrArticle = self.ArticleList.pop(0)
            for sentence in self.CurrArticle.sentences:
                for word in sentence.words:
                    self.CurrArtWordList.append(word)
        else:
            self.CurrArticle == None

    def _nextWord(self):
        if len(self.CurrArtWordList) == 0:
            self._nextArticle()
        if len(self.CurrArtWordList) == 0:
            return None
        return self.CurrArtWordList.pop(0)

    def _nextSentence(self):
        if (self.CurrArticle is None) or (len(self.CurrArticle.sentences) == 0):
            self._nextArticle()
        if self.CurrArticle is None or (len(self.CurrArticle.sentences) == 0):
            return None
        return self.CurrArticle.sentences.pop(0)

    def GetResultWordList(self):
        wordList = []
        for index in xrange(self.length):
            wordItem = self.reverse_dictionary[index + 1]
            wordItem["vector"] = self.vectors[index, :].tolist()
            del wordItem["_id"]
            wordList.append(wordItem)
        return wordList
