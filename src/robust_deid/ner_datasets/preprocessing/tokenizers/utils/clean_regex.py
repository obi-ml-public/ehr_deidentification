from typing import List
class CleanRegex(object):
    """
    This class is used to define the regexes that will be used by the
    spacy tokenizer rules. Mainly the regexes are used to clean up
    tokens that have unwanted characters (e.g extra hyphens).
    """
    #Staff - 3
    #Hosp - 4, 5
    #Loc - 2
    @staticmethod
    def get_prefixes() -> List[str]:
        """
        This function is used to build the regex that will clean up dirty characters
        present at the prefix position (start position) of a token. For example the token ---clean
        has three hyphens that need to be split from the word clean. This regex
        will be used by spacy to clean it up. This rule considers any characters that is
        not a letter or a digit as dirty characters
        Examples: ----------------9/36, :63, -ESH
        Returns:
            (list): List of regexes to clean the prefix of the token
        """
        #Handles case 5 of HOSP
        return ['((?P<prefix>([^a-zA-Z0-9.]))(?P=prefix)*)', '([.])(?!\d+(\W+|$))']
    
    @staticmethod
    def get_suffixes() -> List[str]:
        """
        This function is used to build the regex that will clean up dirty characters
        present at the suffix position (end position) of a token. For example the token clean---
        has three hyphens that need to be split from the word clean. This regex
        will be used by spacy to clean it up. This rule considers any characters that is
        not a letter or a digit as dirty characters
        Examples: FRANK^, regimen---------------, no)
        Returns:
            (list): List of regexes to clean the suffix of the token
        """
        return ['((?P<suffix>([^a-zA-Z0-9]))(?P=suffix)*)']
    
    @staticmethod
    def get_infixes() -> List[str]:
        """
        This function is used to build the regex that will clean up dirty characters
        present at the infix position (in-between position) of a token. For example the token 
        clean---me has three hyphens that need to be split from the word clean and me. This regex
        will be used by spacy to clean it up. This rule considers any characters that is
        not a letter or a digit as dirty characters
        Examples: FRANK^08/30/76^UNDERWOOD, regimen---------------1/37
        Returns:
            (list): List of regexes to clean the infix of the token
        """
        #Handles case 3 of STAFF
        #Handles case 4 of HOSP
        #Handles case 2 of LOC
        connector_clean = '\^|;|&#|([\(\)\[\]:="])'
        #full_stop_clean = '(?<=[a-zA-Z])(\.)(?=([A-Z][A-Za-z]+)|[^a-zA-Z0-9_.]+)'
        bracket_comma_clean = '(((?<=\d)[,)(](?=[a-zA-Z]+))|((?<=[a-zA-Z])[,)(](?=\w+)))'
        #special_char_clean = '(?<=[a-zA-Z])(\W{3,}|[_]{3,})(?=[A-Za-z]+)'
        special_char_clean = '(?<=[a-zA-Z])([_\W_]{3,})(?=[A-Za-z]+)'
        #Sometimes when there is no space between a period and a comma - it becomes part of the same token
        #e.g John.,M.D - we need to split this up.
        comma_period_clean = '(?<=[a-zA-Z])(\.,)(?=[A-Za-z]+)'
        
        return [connector_clean, bracket_comma_clean, special_char_clean, comma_period_clean]