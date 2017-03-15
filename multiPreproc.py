#!coding: utf-8
import time
from langPreprocesser import *
import chardet
import Util
import threading

hasText = True
mutex = threading.Lock()
writeMutex = threading.Lock()
RawArticleList = []

def write2file(preprocesser, outfile):
    global writeMutex
    # write data to file
    if writeMutex.acquire():
        preprocesser.Write2File(outfile)
        writeMutex.release()
    # update word frequence
    for key in preprocesser.wordDict:
        Util.UpdateWordFreg(key, preprocesser.wordDict[key])
    # clean articles
    preprocesser.Clean()

def procActicle(outfile):
    global hasText, RawArticleList, mutex, writeMutex
    preprocesser = ChinesePreprocesser();
    while hasText or (len(RawArticleList) > 0):
        rawArticle = None
        if mutex.acquire(1):
            if len(RawArticleList) > 0:
                rawArticle = RawArticleList.pop(0)
            mutex.release()
        if (rawArticle is not None) and (len(rawArticle) > 0):
            article = Article()
            for line in rawArticle:
                article.FeedRawContent(line)
            preprocesser.ArticleList.append(article)
            if len(preprocesser.ArticleList) > 20:
                write2file(preprocesser, outfile)
    # clean last article
    write2file(preprocesser, outfile)

input_file = open("data/wiki_cn_03", "r")
output_file = open("wiki_cn_format_03", "w")
threadList = []
for i in range(4):
    t = threading.Thread(target=procActicle, args=(output_file, ))
    t.start()
    threadList.append(t)
index = 0
rawArticle = []
for line in input_file:
    print(index)
    index += 1
    while len(RawArticleList) > 100:
        time.sleep(1)
    if Util.IsValidForWiki(line):
        rawArticle.append(line)
    elif line == "</doc>\n":
        if mutex.acquire():
            RawArticleList.append(rawArticle)
            mutex.release()
        rawArticle = []
hasText = False
for t in threadList:
    t.join()
