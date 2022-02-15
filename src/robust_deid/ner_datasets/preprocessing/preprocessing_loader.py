from typing import Union, Optional, Sequence

from .sentencizers import SpacySentencizer, NoteSentencizer
from .tokenizers import ClinicalSpacyTokenizer, SpacyTokenizer, CoreNLPTokenizer


class PreprocessingLoader(object):

    @staticmethod
    def get_sentencizer(sentencizer: str) -> Union[SpacySentencizer, NoteSentencizer]:
        """
        Get the desired the sentencizer
        We can either use the sci-spacy (en_core_sci_lg or en_core_web_sm) or
        consider the entire note as a single sentence.
        Args:
            sentencizer (str): Specify which sentencizer you want to use
        Returns:
            Union[SpacySentencizer, NoteSentencizer]: An object of the requested
                                                      sentencizer class
        """
        if sentencizer in ['en_core_sci_lg', 'en_core_sci_md', 'en_core_sci_sm', 'en_core_web_sm']:
            return SpacySentencizer(spacy_model=sentencizer)
        elif sentencizer == 'note':
            return NoteSentencizer()
        else:
            raise ValueError('Invalid sentencizer - does not exist')

    @staticmethod
    def get_tokenizer(tokenizer: str) -> Union[SpacyTokenizer, ClinicalSpacyTokenizer, CoreNLPTokenizer]:
        """
        Initialize the tokenizer based on the CLI arguments
        We can either use the default scipacy (en_core_sci_lg or en_core_web_sm)
        or the modified scipacy (with regex rule) tokenizer.
        It also supports the corenlp tokenizer
        Args:
            tokenizer (str): Specify which tokenizer you want to use
        Returns:
            Union[SpacyTokenizer, ClinicalSpacyTokenizer, CoreNLPTokenizer]: An object of the requested tokenizer class
        """
        if tokenizer in ['en_core_sci_lg', 'en_core_sci_md', 'en_core_sci_sm', 'en_core_web_sm', 'en']:
            return SpacyTokenizer(spacy_model=tokenizer)
        elif tokenizer == 'corenlp':
            return CoreNLPTokenizer()
        elif tokenizer == 'clinical':
            # Abbreviations - we won't split tokens that match these (e.g 18F-FDG)
            return ClinicalSpacyTokenizer(spacy_model='en_core_sci_sm')
        else:
            raise ValueError('Invalid tokenizer - does not exist')
