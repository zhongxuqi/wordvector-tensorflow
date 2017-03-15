import os, time
from langPreprocesser.ChinesePreprocesser import *

datafile = open("format_file_out", "r")
articleCount = 0
mLangReader = ChineseReader()
for line in datafile:
    mLangReader.FeedContent(line)
    if line == "</end>\n":
        articleCount += 1
    if articleCount>=200:
        break

print("article count:", articleCount)
wordCount = 0
for article in mLangReader.ArticleList:
    for sentence in article.sentences:
        wordCount += len(sentence.words)
print("words:", wordCount)
time.sleep(3)
