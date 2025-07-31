import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from typing import List, Dict, Optional, Tuple
import logging
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ItalianTextProcessor:
    def __init__(self, load_spacy: bool = True): 
        self.spacy_model = None
        self._setup_nltk()
        if load_spacy and SPACY_AVAILABLE:
            self._setup_spacy()
        self._setup_fallback_lemmatizer()
    def _setup_nltk(self) -> None:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)

        self.stemmer = SnowballStemmer('italian')
        self.stop_words = set(stopwords.words('italian'))
        logger.info("NLTK components initialized")
    def _setup_spacy(self) -> None:
        try:
            self.spacy_model = spacy.load("it_core_news_sm")
            logger.info("spaCy Italian model loaded successfully")
        except OSError:
            logger.warning(
                "spaCy Italian model not found. "
                "Install with: python -m spacy download it_core_news_sm"
            )
            self.spacy_model = None
    def _setup_fallback_lemmatizer(self) -> None:
        self.lemma_dict = {
            'sono': 'essere', 'sei': 'essere', 'è': 'essere', 'siamo': 'essere',
            'siete': 'essere', 'era': 'essere', 'erano': 'essere',
            'ho': 'avere', 'hai': 'avere', 'ha': 'avere', 'abbiamo': 'avere',
            'avete': 'avere', 'hanno': 'avere', 'aveva': 'avere', 'avevano': 'avere',
            'faccio': 'fare', 'fai': 'fare', 'fa': 'fare', 'facciamo': 'fare',
            'fate': 'fare', 'fanno': 'fare', 'faceva': 'fare', 'facevano': 'fare',
            'vado': 'andare', 'vai': 'andare', 'va': 'andare', 'andiamo': 'andare',
            'andate': 'andare', 'vanno': 'andare', 'andava': 'andare', 'andavano': 'andare',
            'dico': 'dire', 'dici': 'dire', 'dice': 'dire', 'diciamo': 'dire',
            'dite': 'dire', 'dicono': 'dire', 'diceva': 'dire', 'dicevano': 'dire',
            'vedo': 'vedere', 'vedi': 'vedere', 'vede': 'vedere', 'vediamo': 'vedere',
            'vedete': 'vedere', 'vedono': 'vedere', 'vedeva': 'vedere', 'vedevano': 'vedere',
            'case': 'casa', 'libri': 'libro', 'bambini': 'bambino', 'bambine': 'bambina',
            'uomini': 'uomo', 'donne': 'donna', 'ragazzi': 'ragazzo', 'ragazze': 'ragazza',
            'gatti': 'gatto', 'cani': 'cane', 'macchine': 'macchina', 'strade': 'strada',
            'bella': 'bello', 'belle': 'bello', 'belli': 'bello',
            'grande': 'grande', 'grandi': 'grande',
            'piccola': 'piccolo', 'piccole': 'piccolo', 'piccoli': 'piccolo',
            'buona': 'buono', 'buone': 'buono', 'buoni': 'buono',
        }

    @property
    def spacy_available(self) -> bool:
        return self.spacy_model is not None

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        tokens = word_tokenize(text.lower(), language='italian')

        tokens = [
            token for token in tokens
            if token not in string.punctuation and token.strip()
        ]

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
            
        if self.spacy_model:
            text = " ".join(tokens)
            doc = self.spacy_model(text)
            return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        else:
            lemmas = []
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.lemma_dict:
                    lemmas.append(self.lemma_dict[token_lower])
                else:
                    lemmas.append(self.stemmer.stem(token_lower))
            return lemmas

    def get_pos_tags(self, text: str) -> List[Tuple[str, str, str]]:
        if not self.spacy_model:
            return []

        doc = self.spacy_model(text.lower())
        return [
            (token.text, token.pos_, token.tag_)
            for token in doc
            if not token.is_punct and not token.is_space
        ]

    def process_text(self, text: str, remove_stopwords: bool = True) -> Dict:
        if not text or not text.strip():
            return {
                'original_text': text,
                'all_tokens': [],
                'filtered_tokens': [],
                'stems': [],
                'lemmas': [],
                'pos_tags': [],
                'total_tokens': 0,
                'filtered_token_count': 0,
                'stopwords_removed': 0,
                'spacy_available': self.spacy_available
            }

        tokens = self.tokenize(text)
        original_tokens = tokens.copy()
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        stems = self.stem(tokens)

        lemmas = self.lemmatize_tokens(tokens)

        pos_tags = self.get_pos_tags(" ".join(tokens)) if tokens else []

        return {
            'original_text': text,
            'all_tokens': original_tokens,
            'filtered_tokens': tokens,
            'stems': stems,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'total_tokens': len(original_tokens),
            'filtered_token_count': len(tokens),
            'stopwords_removed': len(original_tokens) - len(tokens),
            'spacy_available': self.spacy_available
        }

    def compare_methods(self, text: str) -> Dict:
        tokens = self.remove_stopwords(self.tokenize(text))
        return {
            'original_text': text,
            'tokens': tokens,
            'nltk_stems': self.stem(tokens),
            'spacy_lemmas': self.lemmatize_tokens(tokens),
            'pos_tags': self.get_pos_tags(text) if self.spacy_available else []
        }
