import spacy
from loguru import logger
import pandas as pd
from collections import defaultdict
import re

class Classical:
    """ 
    Class containing methods for classical NLP tasks
    """
    def __init__(self,text,spaCy_model:str = "en_core_web_sm") -> None:
        self.text = text
        if spaCy_model == "en_core_web_sm" or "en_core_web_md" or "en_core_web_trf" or "en_core_web_lg":
            self.spaCy_model = spaCy_model 
            self.nlp = spacy.load(spaCy_model)
        else :
            logger.error("Invalid spaCy model")
        self.stat_dict = self.stats_dict()
        self.doc = self.tokenize()
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
    def word_counter_df(self):
        """
        Count words in text and return as dataframe
        """
        word_counter = self.word_counter()
        word_counter_df = pd.DataFrame.from_dict(word_counter,orient="index",columns=["frequency"])
        logger.info("Word counter dataframe generated")
        return word_counter_df
    def regex_search(self,pattern:str):
        """
        Search text with regex
        """
        regex_search = re.findall(pattern,self.text)
        logger.info("Regex search completed")
        return regex_search
    def get_named_entities(self):
        """
        Get named entities from text
        """
        named_entities = [(ent.text, ent.label_) for ent in self.doc.ents]
        logger.info("Named entities extracted")
        return named_entities
    
    def spacy_matcher(self,pattern_name:str,PATTERN:list):
        """
        Use spaCy matcher to match patterns
        Example: print(genesis.spacy_matcher(pattern_name="God",PATTERN=[{"TEXT":"God"}]))
        """
        matcher = spacy.matcher.Matcher(self.nlp.vocab)
        matcher.add(pattern_name,[PATTERN])
        matches = matcher(self.doc)
        logger.info(f"{len(matches)} matches found")
        match_dict = {}
        count = 0
        for match_id, start, end in matches:
            match_dict[count] = (start,end,self.doc[start:end].text)
            count += 1
        logger.info("spaCy matcher completed")
        return match_dict
            
    def spacy_compare(self,external_doc):
        """
        Compare text with external document
        """
        external_doc = self.nlp(external_doc)
        similarity = self.doc.similarity(external_doc)
        logger.info("Text compared")
        return similarity