from typing import List
from tqdm import tqdm
from abc import ABC, abstractmethod
from Levenshtein import distance as levenshtein_distance


class BaseSlurExtractor(ABC):
    """ Базовый класс для извлечения матов из текста.
    Используется для создания различных моделей извлечения матов.
    """
    def __init__(self, verbose: bool=True):
        self.is_fitted = False
        self.verbose = verbose
    
    @abstractmethod
    def fit(self, texts: List[str], slurs: List[str]):
        """Обучение на данных"""
        pass
    
    @abstractmethod
    def transform(self, text: List[str]) -> List[str]:
        """Извлечение матов из текста. 
        Возвращает список строк вида 'слово1,слово2'
        """
        pass
    
    def fit_transform(self, texts: List[str], slurs: List[str]) -> List[str]:
        """Комбо fit + transform"""
        self.fit(texts, slurs)
        return self.transform(texts)
    
    def evaluate(self, texts: List[str]=None, true_slurs: List[str]=None, 
                 predicts: List[str]=None, name="Model") -> float:
        """Оценка модели по Левенштейну"""
        predictions = predicts if predicts else self.transform(texts)
        if true_slurs is None:
            raise ValueError("true_slurs must be provided for evaluation.")
        
        total_distance = 0
        if self.verbose:
            iterator = tqdm(zip(predictions, true_slurs), desc=f"Evaluating {name}", total=len(predictions))
        else:
            iterator = zip(predictions, true_slurs)
        
        for pred, true in iterator:
            total_distance += levenshtein_distance(pred, str(true))
        
        avg_distance = total_distance / len(predictions)
        if self.verbose:
            print(f"{name} - Average Levenshtein Distance: {avg_distance:.4f}")
        return avg_distance