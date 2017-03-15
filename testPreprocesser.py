#!coding: utf-8
import codecs
from langPreprocesser import ChinesePreprocesser
import chardet
import Util
import re

# preprocesser = ChinesePreprocesser();
# preprocesser.FeedSentence(["a"])
#
# preprocesser.FeedRawContent("我们在")
# preprocesser.FeedRawContent("而执行内容可以多行")
# datafile = open("data/wiki_cn_00","r")
# datafile.readline()
# datafile.readline()
# datafile.readline()
# preprocesser.FeedRawContent(datafile.readline())
# print("out:")
# preprocesser.ShowContent()
# datafile.close()
# sent = "当需要在条件不成《》立时\"从条件不成立\"iu结局infr“疯狂就热风”,fdsf执行内容则fsdf可以执行相关语句《范德萨空间和疯狂》"
# print(re.findall("(《.+》|\".+\"|“.+”)", sent))

RawSubItem = ["[^\u4e00-\u9fa5a-zA-Z0-9]+", "[\u4e00-\u9fa5]+", "[a-zA-Z0-9]+"]
sent = "当需要在条件不djfdiuf8787范德DGFDS萨3248,fdfr34"
print(re.findall("(" + "|".join(RawSubItem) + ")", sent))

# sentblocks = []
# lastEnd = 0
# for match in re.compile("(《.*?》|\".*?\"|“.*?”)").finditer(sent):
#     if match.start() > lastEnd:
#         sentblocks.append(sent[lastEnd:match.start()])
#     sentblocks.append(sent[(match.start() + 1):(match.end() - 1)])
#     lastEnd = match.end()
# if lastEnd < len(sent):
#     sentblocks.append(sent[lastEnd:])
# print(sentblocks)

# if match:
#     print(match.group())

# preprocesser = ChinesePreprocesser();
# datafile = open("data/wiki_cn_00","r")
# outfile = open("format_file_out","w+")
# articleCount = 0
# index = 0
# for line in datafile.readlines():
#     print(index)
#     index += 1
#     if Util.IsValidForWiki(line):
#         preprocesser.FeedRawContent(line)
#     elif line == "</doc>\n":
#         preprocesser.NewActicle()
#         articleCount += 1
#         if articleCount % 100 == 0:
#             preprocesser.Write2File(outfile)
#             for key in preprocesser.wordDict:
#                 Util.UpdateWordFreg(key, preprocesser.wordDict[key])
#             preprocesser.Clean()
#
# preprocesser.Write2File(outfile)
# for key in preprocesser.wordDict:
#     Util.UpdateWordFreg(key, preprocesser.wordDict[key])
# preprocesser.Clean()
