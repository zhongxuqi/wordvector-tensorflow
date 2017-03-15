import Util
import chardet
import re

class BaseVocabularyMachine:
    TIMES = "times"
    TimesLimit = 10

    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = {}
        self.length = 0
        self.sentences = []

    def FeedWords(self, words, all_accept = False):
        for wordItem in words:
            if (all_accept or \
            wordItem[BaseVocabularyMachine.TIMES] >= BaseVocabularyMachine.TimesLimit) \
            and (wordItem["word"] not in self.dictionary):
                self.length += 1
                wordItem["index"] = self.length
                self.dictionary[wordItem["word"]] = wordItem
                self.reverse_dictionary[self.length] = wordItem

class Sentence:
    ChineseBlock = "[\u4e00-\u9fa5]+"
    RawSubItem = ["[^\u4e00-\u9fa5a-zA-Z0-9]+", "[\u4e00-\u9fa5]+", "[a-zA-Z0-9]+"]
    ForceBlock = ["《.*?》", "\".*?\"", "“.*?”", "<.*?>", "'.*?'"]

    def __init__(self):
        self.words = []

    def initSentence(self, content):
        blockPattern = "(" + "|".join(Sentence.ForceBlock) + ")"
        lastEnd = 0
        for match in re.compile(blockPattern).finditer(content):
            if match.start() > lastEnd:
                self.splitWord(content[lastEnd:match.start()])
            if match.start() + 2 > match.end():
                self.words.append(content[(match.start() + 1):(match.end() - 1)])
            lastEnd = match.end()
        if lastEnd < len(content):
            self.splitWord(content[lastEnd:])


    def splitWord(self, content_block):
        pattern = "(" + "|".join(Sentence.RawSubItem) + ")"
        SubSentences = re.findall(pattern, content_block)
        for subItem in SubSentences:
            if self.isChineseBlock(subItem):
                subWords = Sentence.splitWordsBlock(subItem)
                for subsubItem in subWords:
                    self.words.append(subsubItem)
            else:
                self.words.append(subItem)

    def FeedFormatSentence(self, formatSentence):
        for word in formatSentence.split(" "):
            if (len(word) > 0) and (word != "\n"):
                self.words.append(word)

    def isChineseBlock(self, block):
        m = re.search(Sentence.ChineseBlock, block)
        if m:
            return True
        else:
            return False

    def splitWordsBlock(block):
        subWords = []
        prevWord = [0, 0]
        indexCurr = 0
        clipSecondWordLen = 0
        while indexCurr < len(block):
            index = indexCurr
            # match word
            for i in range(4):
                if Util.IsWord(block[indexCurr:(indexCurr + 1 + i)]):
                    index = indexCurr + 1 + i

            # if not match, match reversely
            if (index == indexCurr) and (indexCurr > 0) and prevWord[1] == indexCurr:
                backOff = 1
                while indexCurr - backOff > prevWord[0]:
                    indexClip = indexCurr - backOff
                    for clipSecondWordLen in range(backOff,4):
                        if Util.IsWord(block[indexClip:(indexClip + 1 + clipSecondWordLen)]) \
                        or Util.IsWord(block[prevWord[0]:indexClip]):
                            index = indexClip
                            break
                    if index != indexCurr:
                        break
                    backOff += 1

            # check the match result
            if index > indexCurr:
                # append lost matched words
                if indexCurr > prevWord[1]:
                    subWords.append(block[prevWord[1]:indexCurr])
                    # It maybe a word. So store it
                    Util.StoreMaybeWord(block[prevWord[1]:indexCurr])
                prevWord = [indexCurr, index]
                # append matched words
                subWords.append(block[prevWord[0]:prevWord[1]])
                indexCurr = index
            elif index < indexCurr:
                # revise wrong matched words
                subWords[-1] = block[prevWord[0]:index]
                # append matched words
                subWords.append(block[index:(index + clipSecondWordLen + 1)])
                prevWord = [index, index + clipSecondWordLen + 1]
                indexCurr = index + clipSecondWordLen + 1
            else:
                indexCurr += 1
                # append lost matched words
                if indexCurr >= len(block):
                    subWords.append(block[prevWord[1]:indexCurr])
                    # It maybe a word. So store it
                    Util.StoreMaybeWord(block[prevWord[1]:indexCurr])
        return subWords

    def toSpiltLine(self):
        line = ""
        for word in self.words:
            line = line + word + " "
        return line

class Article:
    ArticleEnd = "</end>\n"
    SentenceEnd = ["\\n", "。", ".", "!", "！", ";", "；", "?", "？"]

    def __init__(self):
        self.sentences = []

    def FeedContent(self, content):
        pattern = "[" + "".join(Article.SentenceEnd) + "]+"
        rawSententces = re.compile(pattern).split(content)
        for item in rawSententces:
            if Util.HasChineseWord(item):
                sentence = Sentence()
                sentence.initSentence(item)
                self.sentences.append(sentence)

    def FeedRawContent(self, rawContent):
        if not isinstance(rawContent, str):
            raise Exception("input is not str")
        # decode rawContent
        content = ""
        try:
            b = bytes(rawContent)
            encode = chardet.detect(b)
            content = b.decode(encode)
        except Exception as e:
            content = rawContent

        # init article
        self.FeedContent(content)

    def FeedFormatSentence(self, format_sentence):
        SentenceObject = Sentence()
        SentenceObject.FeedFormatSentence(format_sentence)
        self.sentences.append(SentenceObject)

    def ToString(self):
        outContent = ""
        for sentence in self.sentences:
            outContent = outContent + sentence.toSpiltLine() + "\n"
        return outContent
