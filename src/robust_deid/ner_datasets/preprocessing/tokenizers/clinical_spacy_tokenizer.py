import re
import spacy
from spacy.symbols import ORTH
from .spacy_tokenizer import SpacyTokenizer
from .utils import DateRegex, CleanRegex, ClinicalRegex

def read_abbreviations():
    import importlib.resources as pkg_resources
    from . import abbreviations
    abbrevs = []
    with pkg_resources.open_text(abbreviations, 'medical_abbreviations.txt') as f:
        abbrevs += [line.rstrip('\n') for line in f]
    return abbrevs

class ClinicalSpacyTokenizer(SpacyTokenizer):
    """
    This class is used to read text and return the tokens
    present in the text (and their start and end positions)
    """

    def __init__(
            self,
            spacy_model,
            split_multiple=True,
            split_temperature=True,
            split_percentage=True
    ):
        """
        Initialize a spacy model to read text and split it into 
        tokens.
        Args:
            spacy_model (str): Name of the spacy model
        """
        super().__init__(spacy_model)
        self._nlp.tokenizer.prefix_search = self.__get_prefix_regex(split_multiple, split_temperature,
                                                                    split_percentage).search
        self._nlp.tokenizer.infix_finditer = self.__get_infix_regex().finditer
        self._nlp.tokenizer.suffix_search = self.__get_suffix_regex().search
        new_rules = {}
        for orth, exc in self._nlp.tokenizer.rules.items():
            if re.search('((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[.]$)|(^(W|w)ed$)', orth):
                continue
            new_rules[orth] = exc
        self._nlp.tokenizer.rules = new_rules
        abbreviations = read_abbreviations()
        for abbreviation in abbreviations:
            special_case = [{ORTH: abbreviation}]
            self._nlp.tokenizer.add_special_case(abbreviation, special_case)
        # this matches any lower case tokens - abstract this part out - whetehr to lowercase abbreviations ro not
        exclusions_uncased = {abbreviation.lower(): [{ORTH: abbreviation.lower()}] for abbreviation in
                              abbreviations}
        for k, excl in exclusions_uncased.items():
            try:
                self._nlp.tokenizer.add_special_case(k, excl)
            except:
                print('failed to add exception: {}'.format(k))

    def __get_prefix_regex(self, split_multiple, split_temperature, split_percentage):

        date_prefix = DateRegex.get_infixes()
        clinical_prefix = ClinicalRegex.get_prefixes(split_multiple, split_temperature, split_percentage)
        clean_prefix = CleanRegex.get_prefixes()
        digit_infix = ClinicalRegex.get_infixes()
        prefixes = clean_prefix + self._nlp.Defaults.prefixes + date_prefix + clinical_prefix + digit_infix
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        return prefix_regex

    def __get_suffix_regex(self):
        clean_suffix = CleanRegex.get_suffixes()
        suffixes = clean_suffix + self._nlp.Defaults.suffixes
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        return suffix_regex

    def __get_infix_regex(self):

        date_infixes = DateRegex.get_infixes()
        clean_infixes = CleanRegex.get_infixes()
        digit_infix = ClinicalRegex.get_infixes()
        infixes = self._nlp.Defaults.infixes + date_infixes + clean_infixes
        infix_re = spacy.util.compile_infix_regex(infixes)
        return infix_re

    def get_nlp(self):
        return self._nlp
