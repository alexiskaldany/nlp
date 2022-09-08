import spacy
from loguru import logger
import pandas as pd
from collections import defaultdict

class Classical:
    """ 
    Class containing methods for classical NLP tasks
    """
    def __init__(self,text,spaCy_model:str = "en_core_web_sm") -> None:
        self.text = text
        if spaCy_model == "en_core_web_sm" or "en_core_web_md" or "en_core_web_trf":
            self.spaCy_model = spaCy_model 
            self.nlp = spacy.load(spaCy_model)
        self.stat_dict = self.stats_dict()
        logger.info("Classical NLP object created")
        pass
    
    def tokenize(self):
        """
        Tokenize text
        """
        doc = self.nlp(self.text)
        logger.info("Text tokenized")
        return doc
    def stats_dict(self):
        """
        Generate text summary
        """
        stat_dict = {}
        stat_dict["text_length"] = len(self.text)
        stat_dict["text_sentences"] = len(list(self.nlp(self.text).sents))
        stat_dict["text_words"] = len(self.text.split(" "))
        stat_dict["text_lines"] = len(self.text.splitlines())
        stat_dict["text_unique_words"] = len(set(self.text.split(" ")))
        stat_dict["text_unique_words_percentage"] = round((stat_dict["text_unique_words"]/stat_dict["text_words"])*100,2)
        logger.info("Text summary generated")
        return stat_dict
    def word_counter(self):
        """
        Count words in text
        """
        word_counter = defaultdict(int)
        for word in self.text.split(" "):
            if word in word_counter:
                word_counter[word] += 1
            else:
                word_counter[word] = 1
        word_counter = {k: v for k, v in sorted(word_counter.items(), key=lambda item: item[1],reverse=True)}
        logger.info("Word counter generated")
        return word_counter