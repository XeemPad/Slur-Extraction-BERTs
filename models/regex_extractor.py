import re
from typing import List, Set

import numpy as np
import pandas as pd

from tqdm import tqdm
from models.base_extractor import BaseSlurExtractor


class RegexExtractor(BaseSlurExtractor):    
    def __init__(self, dictionary: Set[str], **kwargs):
        super().__init__(**kwargs)
        self.dictionary = dictionary
        
        # Символы маскировки
        self.mask_chars = r'[*.-_@#$%]'

    def fit(self, texts: pd.Series, slurs: pd.Series):
        # Создаем паттерны для каждого слова из словаря
        self.patterns = self._create_patterns()

        self.is_fitted = True
        return self
    
    def transform(self, texts, n_jobs=10) -> List[str]:
        """Применяет regex поиск к списку текстов"""
        if self.verbose:
            found_slurs = [self._find_slurs_in_text(text) for text in tqdm(texts, desc="Transforming texts with regex")]
        else:
            found_slurs = texts.map(self._find_slurs_in_text)
        
        return np.array(found_slurs, dtype=object)
    
    def _create_patterns(self) -> List[re.Pattern]:
        """Создает regex паттерны для всех слов из словаря"""
        patterns = []
        
        for word in tqdm(self.dictionary, desc="Fit: creating patterns"):
                
            # Создаем различные варианты маскировки
            word_patterns = []
            
            # 1. Точное совпадение
            word_patterns.append(re.escape(word))
            
            # 2. Замена гласных на маски
            vowels = 'аеёиоуыэюя'
            vowel_masked = word
            for vowel in vowels:
                vowel_masked = vowel_masked.replace(vowel, f'[{vowel}{self.mask_chars}]')
            word_patterns.append(vowel_masked)
            
            # 3. Замена средних букв на маски (оставляем первую и последнюю)
            if len(word) > 3:
                masked_middle = word[0] + f'{self.mask_chars}+' + word[-1]
                word_patterns.append(masked_middle)
            
            # 4. Замена отдельных букв на маски
            for i in range(1, len(word) - 1):
                char_masked = word[:i] + f'{self.mask_chars}' + word[i+1:]
                word_patterns.append(char_masked)
            
            # 5. Разделение пробелами/символами
            spaced = r'\s*'.join(list(word))
            word_patterns.append(spaced)
            
            # Компилируем паттерны
            for pattern in word_patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    patterns.append((compiled, word))
                except re.error:
                    continue  # Пропускаем некорректные паттерны
        
        return patterns
    
    def _find_slurs_in_text(self, text: str) -> str:
        """Находит маты в тексте используя regex паттерны"""
        text_lower = text.lower()
        found_words = set()
        
        # Применяем все паттерны
        for pattern, original_word in self.patterns:
            if pattern.search(text_lower):
                found_words.add(original_word)
        
        # Дополнительный поиск с учетом л33т-спика и замен
        found_words.update(self._find_leet_variants(text_lower))
        
        return ','.join(sorted(found_words)) if found_words else ''
    
    def _find_leet_variants(self, text: str) -> Set[str]:
        """Ищет л33т-варианты (a->@, o->0, и т.д.)  ОТКЛЮЧЕНО ДЛЯ УСКОРЕНИЯ"""
        found = set()
        return found
        
        # Словарь замен л33т-спика
        leet_map = {
            'а': '[а@4]', 'о': '[о0]', 'е': '[е3]', 'и': '[и1!]',
            'у': '[уy]', 'р': '[рp]', 'х': '[хx]', 'с': '[сc$]',
            'в': '[вb]', 'н': '[нh]', 'к': '[кk]', 'т': '[тt]'
        }
        
        for word in self.dictionary:
            if len(word) < 3:
                continue
                
            # Создаем л33т-паттерн
            leet_pattern = word
            for char, replacement in leet_map.items():
                leet_pattern = leet_pattern.replace(char, replacement)
            
            try:
                if re.search(leet_pattern, text, re.IGNORECASE):
                    found.add(word)
            except re.error:
                continue
        
        return found
