import numpy as np
import pandas as pd

from typing import Set
from collections import namedtuple

from models.base_extractor import BaseSlurExtractor
from Levenshtein import distance as levenshtein_distance


class DictionaryExtractor(BaseSlurExtractor):
    m_namedtpl = namedtuple('Methods', ['match', 'substr', 'levenshtein'])
    SRCH_METHODS = m_namedtpl(match='match', substr='substr', levenshtein='levenshtein')

    def __init__(self, dictionary: Set[str], method=SRCH_METHODS.match, allowed_dist: float=2, **kwargs):
        super().__init__(**kwargs)
        
        self.dictionary: set|dict = dictionary
        self.is_fitted = True  # Словарь уже готов
        self.method = method  # 'match' or 'substr' or 'levenshtein'
        # Если метод levenshtein, то задаём допустимое расстояние
        self.allowed_distance = allowed_dist
    
    def fit(self, texts: pd.Series, slurs: pd.Series):
        self.is_fitted = True
        return self
    
    def transform(self, texts: pd.Series) -> str:
        # texts = texts.str.lower().str.replace(r'[^\w\s]', '', regex=True)  # Удаляем пунктуацию
        found_slurs = texts.map(lambda text: self._find_slurs_in_text(text))
        return np.array(found_slurs, dtype=object)
   
    def _find_slurs_with_match(self, words: Set[str]) -> Set[str]:
        """Находит точные совпадения слов из словаря в тексте"""
        return words.intersection(self.dictionary)
    
    def _find_slurs_with_substr(self, words: Set[str]) -> Set[str]:
        """Находит слова из словаря, которые являются подстроками в тексте"""
        found_slurs = set()
        for word in words:
            for dict_word in self.dictionary:
                if word in dict_word:
                    found_slurs.add(word)
        return found_slurs
    
    def _find_slurs_with_levenshtein(self, words: Set[str]) -> Set[str]:
        """Находит слова из словаря, которые похожи на слова в тексте по Левенштейну"""
        found_slurs = set()
        for word in words:
            for dict_word in self.dictionary:
                if levenshtein_distance(word, dict_word) <= self.allowed_distance:
                    found_slurs.add(dict_word)
        return found_slurs
    
    def _find_slurs_in_text(self, text: str) -> str:
        """Находит все слова из словаря в тексте с учётом выбранного метода"""
        words = set(text.split())
        
        match self.method:
            case self.SRCH_METHODS.match:
                found_slurs = self._find_slurs_with_match(words)
            case self.SRCH_METHODS.substr:
                found_slurs = self._find_slurs_with_substr(words)
            case self.SRCH_METHODS.levenshtein:
                found_slurs = self._find_slurs_with_levenshtein(words)
            case _:
                raise ValueError(f"Unknown method: {self.method}")
        
        return ','.join(sorted(found_slurs)) if found_slurs else ''