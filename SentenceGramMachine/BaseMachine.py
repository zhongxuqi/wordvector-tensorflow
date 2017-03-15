from langPreprocesser.ChinesePreprocesser import *
from base import *
import numpy as np

class BaseMachine(BaseVocabularyMachine):
    INDEX = "index"
    VECTOR = "vector"
    NCE_NUM = 2

    def __init__(self):
        BaseVocabularyMachine.__init__(self)
        self.sentences = []

    def FeedArticle(self, article):
        for sentence in article.sentences:
            self.sentences.append(sentence)

    def FeedArticles(self, articles):
        for article in articles:
            self.FeedArticle(article)

    def nextSentence(self):
        if len(self.sentences) > 0:
            return self.sentences.pop()
        return None

    def nextTrainSentences(self):
        sentence = self.nextSentence()
        sentenceList = []
        if sentence is not None:
            wordList = []
            for word in sentence.words:
                if not Util.IsWord(word) and len(wordList) > 0:
                    sentenceList.append(wordList)
                    wordList = []
                    continue
                if word in self.dictionary:
                    wordList.append(self.dictionary[word][BaseMachine.INDEX])
                elif Util.IsWord(word):
                    wordList.append(0)
            if len(wordList) > 0:
                sentenceList.append(wordList)
                wordList = []
        return sentenceList
