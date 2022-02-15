from typing import List
class ClinicalRegex(object):
    """
    This class is used to define the regexes that will be used by the
    spacy tokenizer rules. Mainly the regexes are used to clean up
    tokens that have unwanted characters and typos (e.g missing spaces).
    In the descriptions when we mention symbol we refer to any character
    that is not a letter or a digit or underscore. The spacy tokenizer splits
    the text by whitespace and applies these rules (along with some default rules)
    to the indiviudal tokens. 
    """
    #Patient - 2, 3, 5
    #Staff - 1, 2
    #Hosp - 2, 3
    #Loc - 1, 3
    @staticmethod
    def get_word_typo_prefix():
        """
        If token contains a typo. What we mean by a typo is when two tokens
        that should be separate tokens are fused into one token because there
        is a missing space.
        Examples: JohnMarital Status - John is the name that is fused into the
        token Marital because of a missing space.
        The regex checks if we have a sequence of characters followed by another
        sequence of characters that starts with a capital letter, followed by two or
        more small letters, we assume this is a typo and split the tokens (two sequences) up.
        If there is a symbol separating the two sequences, we ease the condition saying
        the Cpaital letter can be followed by two or more capital/small letters.
        Returns:
            (str): regex to clean tokens that are fused because of a missing space
        """
        #Handles cases 2 of PATIENT
        #Handles cases 1 & 2 of STAFF
        #Handles cases 2 & 3 of HOSP
        #Handles cases 1 & 3 of LOC
        #'(([a-z]+)|([A-Z]+)|([A-Z][a-z]+))(?=(([-./]*[A-Z][a-z]{2,})|([-./]+[A-Z][a-zA-Z]{2,})))'
        return '(([a-z]+)|([A-Z]{2,})|([A-Z][a-z]+))(?=(([-./]*[A-Z][a-z]{2,})|([-./]+[A-Z][a-zA-Z]{2,})))'
    
    @staticmethod
    def get_word_symbol_digit_prefix() -> str:
        """
        If text is followed by one or more symbols and then followed by one or more digits
        we make the assumption that the text is a seperate token. Spacy will use this regex
        to extract the text portion as one token and will then move on to
        process the rest (symbol and tokens) based on the defined rules.
        Examples: Yang(4986231) - "Yang" will become a seperate token & "(4986231)" will
        be processed as new token
        Returns:
            (str): regex to clean text followed by symbols followed by digits
        """
        #Handles cases 3 & 5 of patient
        return '([a-zA-Z]+)(?=\W+\d+)'
    
    @staticmethod
    def get_multiple_prefix(split_multiple: bool) -> str:
        """
        If text is of the format take it x2 times, this function
        can be used to treat the entire thing as one token or 
        split into two seperate tokens
        Args:
            split_multiple (bool): whether to treat it as one token or split them up
        Returns:
            (str): regex to either keep as one token or split into two
        """
        if(split_multiple):
            return '([x])(?=(\d{1,2}$))'
        else:
            return '[x]\d{1,2}$'

    @staticmethod
    def get_pager_prefix():
        return '([pXxPb])(?=(\d{4,}|\d+[-]\d+))'

    @staticmethod
    def get_age_word_prefix():
        return '([MFmf])(?=\d{2,3}(\W+|$))'
    
    @staticmethod
    def get_id_prefix():
        return '(ID|id|Id)(?=\d{3,})'
    
    @staticmethod
    def get_word_period_prefix():
        return '((cf|CF|Cf|dr|DR|Dr|ft|FT|Ft|lt|LT|Lt|mr|MR|Mr|ms|MS|Ms|mt|MT|Mt|mx|MX|Mx|ph|PH|Ph|rd|RD|Rd|st|ST|St|vs|VS|Vs|wm|WM|Wm|[A-Za-z]{1})[.])(?=((\W+|$)))'
        
    @staticmethod   
    def get_chemical_prefix():
        #Vitamin B12 T9 or maybe codes like I48.9- should probaly do \d{1,2} - limit arbitary numbers
        """
        There are certain chemicals, vitamins etc that should not be split. They 
        should be kept as a single token - for example the token "B12" in
        "Vitamin B12". This regex checks if there is a single capital letter
        followed by some digits (there can be a hyphen in between those digits)
        then this most likely represents a token that should not be split
        Returns:
            (str): regex to keep vitamin/chemical names as a single token
        """
        #return '((\d)?[A-EG-LN-OQ-WYZ]{1}\d+([.]\d+)?(-\d{1,2})*)(?=(([\(\)\[\]:="])|\W*$))'
        return '((\d)?[A-EG-LN-OQ-WYZ]{1}\d+([.]\d+)?(-\d+)*)(?=(([\(\)\[\]:="])|\W*$))'
    
    @staticmethod   
    def get_chemical_prefix_small():
        #Vitamin B12 T9 or maybe codes like I48.9- should probaly do \d{1,2} - limit arbitary numbers
        """
        There are certain chemicals, vitamins etc that should not be split. They 
        should be kept as a single token - for example the token "B12" in
        "Vitamin B12". This regex checks if there is a single capital letter
        followed by some digits (there can be a hyphen in between those digits)
        then this most likely represents a token that should not be split
        Returns:
            (str): regex to keep vitamin/chemical names as a single token
        """
        #return '((\d)?[A-EG-LN-OQ-WYZ]{1}\d+([.]\d+)?(-\d{1,2})*)(?=(([\(\)\[\]:="])|\W*$))'
        return '((\d)?[a-eg-ln-oq-wyz]{1}\d+([.]\d+)?(-\d+)*)(?=(([\(\)\[\]:="])|\W*$))'
    
    @staticmethod
    def get_instrument_prefix():
        """
        There are cases when there are tokens like L1-L2-L3, we want to keep these as one
        single token. This regex checks if there is a capital letter
        Returns:
            (str): regex to keep vitamin/chemical names as a single token
        """
        return '([A-Z]{1,2}\d+(?P<instrument>[-:]+)[A-Z]{1,2}\d+((?P=instrument)[A-Z]{1,2}\d+)*)'
    
    @staticmethod
    def get_instrument_prefix_small():
        """
        There are cases when there are tokens like L1-L2-L3, we want to keep these as one
        single token. This regex checks if there is a capital letter
        Returns:
            (str): regex to keep vitamin/chemical names as a single token
        """
        return '([a-z]{1,2}\d+(?P<instrument_small>[-:]+)[a-z]{1,2}\d+((?P=instrument_small)[a-z]{1,2}\d+)*)'
    
    #Handles Case 3, 4, 5 of MRN
    #Handles Case 1, 2, 3 of PHONE
    #Handles Case 7, 10 of AGE
    #Handles Case 1 of IDNUM
    #Handles Case 3, 5 of PATIENT
    #Handles Case 7 of HOSP
    #Handles Case 1 of General
    @staticmethod
    def get_age_typo_prefix():
        """
        There are cases when there is no space between the text and the age
        Example: Plan88yo - we want Plan to be a seperate token
        Returns:
            (str): 
        """
        age_suffix = '(([yY][eE][aA][rR]|[yY][oO]' + \
        '|[yY][rR]|[yY]\.[oO]|[yY]/[oO]|[fF]|[mM]|[yY])' + \
        '(-)*([o|O][l|L][d|D]|[f|F]|[m|M]|[o|O])?)'
        return '([a-zA-Z]+)(?=((\d{1,3})' + age_suffix + '$))'
    
    @staticmethod
    def get_word_digit_split_prefix():
        #Word followed by more than 3 digits - might not be part of the same token
        #and could be a typo
        #This need not be true - maybe we have an id like BFPI980801 - this will be split
        #BFPI 980801 - but it might be okay to split - need to check
        #([A-Z][a-z]{2,})(?=\d+)
        return '([A-Z][a-z]{2,})(?=[A-Za-z]*\d+)'
    
    @staticmethod
    def get_word_digit_mix_prefix():
        #Mix of letters and characters - most likely a typo if the
        #following characters is a capital letter followed by more than
        #2 small letters
        #return '([A-Z]+\d+([A-Z]+(?!([a-z]{2,}))))(?=(\W+|([A-Z][a-z]{2,})|[a-z]{3,}))'
        return '([A-Z]+\d+)(?=(\W+|([A-Z][a-z]{2,})|[a-z]{3,}))'
    
    @staticmethod
    def get_word_digit_mix_prefix_small():
        #Mix of letters and characters - most likely a typo if the
        #following characters is a capital letter followed by more than
        #2 small letters
        return '([a-z]+\d+)(?=(\W+|[A-Z][a-z]{2,}|[A-Z]{3,}))'
    
    @staticmethod
    def get_word_id_split_prefix():
        return '([a-zA-Z]+)(?=(\d+[-./]+(\d+|$)))'
    
    @staticmethod
    def get_word_section_prefix():
        #Fix JOHNID/CC - missing space from previous section - JOHN
        return '([A-Za-z]+)(?=(((?P<slash>[/:]+)[A-Za-z]+)((?P=slash)[A-Za-z]+)*\W+\d+))'
    
    @staticmethod
    def get_colon_prefix():
        #Split tokens before and after the token
        #Does not split time - we make sure the token ebfore the colon
        #starts with a letter.
        #Splits patterns like <CHAR 1>:<CHAR 2> where CHAR 1 starts with a
        #letter and is followed by one more letters/digits
        #CHAR 2 is a combination of letters/digits of length greater than 2
        #This wont split time, but assumes that when the colon is present
        #the entities on either side of the token are different tokens
        #A:9 - not split - more likely this makes sense as a single token (could be a chemical)
        return '([A-Za-z][A-Za-z0-9]+)(?=([:][A-Za-z0-9]{2,}))'
        
    @staticmethod
    def get_temperature_prefix(split_temperature):
        if(split_temperature):
            return '((\d+)|(\d+[.]\d+))(?=(\u00B0([FCK]{1}|$)))'
        else:
            return '(((\d+)|(\d+[.]\d+))\u00B0([FCK]{1}|$))|(\u00A9[FCK]{1})'
    
    @staticmethod
    def get_percentage_prefix(split_percentage):
        """
        If text is of the format take it 20% times, this function
        can be used to treat the entire thing as one token or 
        split into two seperate tokens
        Args:
            split_percentage (bool): whether to treat it as one token or split them up
        Returns:
            (str): regex to either keep as one token or split into two
        """
        if(split_percentage):
            return '(((\d+)|(\d+[.]\d+)))(?=(%(\W+|$)))'
        else:
            return '(((\d+)|(\d+[.]\d+))%(\W+|$))'
    
    @staticmethod
    def get_value_range_prefixes():
        #The following regex might not work on .4-.5 - no number before decimal point
        #need to figure this out without breaking anything else
        value_range_1 = '(\d{1})(?=([-]((\d{1,2}|(\d+)[.](\d+)))([a-zA-Z]+|[\W]*$)))'
        value_range_2 = '(\d{2})(?=([-]((\d{2,3}|(\d+)[.](\d+)))([a-zA-Z]+|[\W]*$)))'
        value_range_3 = '(\d{3})(?=([-]((\d{3}|(\d+)[.](\d+)))([a-zA-Z]+|[\W]*$)))'
        return value_range_1, value_range_2, value_range_3
    
    @staticmethod
    def get_year_range_prefix():
        return '(\d{4})(?=([-](\d{4})([a-zA-Z]+|[\W]*$)))'
    
    @staticmethod
    def get_short_digit_id_prefix():
        #4A, 3C etc
        return '(\d{1,2}[A-EG-LN-WZ]{1}(?=(\W+|$)))'
    
    #Handles Case 1, 2 of MRN
    #Handles Case 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19 of AGE
    #Handles Case 2, 3, 5 of IDNUM
    #Handles Case 1 of HOSP
    @staticmethod
    def get_digit_symbol_word_prefix():
        return '((\d+)|(\d+[.]\d+))(?=\W+[a-zA-Z]+)'
    
    @staticmethod
    def get_digit_age_split_prefix():
        age_suffix = '(([yY][eE][aA][rR]|[yY][oO]' + \
        '|[yY][rR]|[yY]\.[oO]|[yY]/[oO]|[fF]|[mM]|[yY])' + \
        '(-)*([o|O][l|L][d|D]|[f|F]|[m|M]|[o|O])?)'
        return '((\d{1,3}))(?=(' + age_suffix + '\W*$))'
    
    @staticmethod
    def get_digit_word_short_prefix():
        return '((\d+)|(\d+[.]\d+))([a-z]{1,2}|[A-Z]{1,2})(?=(\W*$))'
        
    @staticmethod
    def get_digit_word_typo_prefix():
        return '((\d+)|(\d+[.]\d+))(?=[a-zA-Z]{1}[a-zA-Z\W]+)'
     
    @staticmethod
    def get_prefixes(split_multiple, split_temperature, split_percentage):
        word_typo_prefix = ClinicalRegex.get_word_typo_prefix()
        word_symbol_digit_prefix = ClinicalRegex.get_word_symbol_digit_prefix()
        pager_prefix = ClinicalRegex.get_pager_prefix()
        age_word_prefix = ClinicalRegex.get_age_word_prefix()
        word_period_prefix = ClinicalRegex.get_word_period_prefix()
        id_prefix = ClinicalRegex.get_id_prefix()
        multiple_prefix = ClinicalRegex.get_multiple_prefix(split_multiple)
        chemical_prefix = ClinicalRegex.get_chemical_prefix()
        chemical_prefix_small = ClinicalRegex.get_chemical_prefix_small()
        instrument_prefix = ClinicalRegex.get_instrument_prefix()
        instrument_prefix_small = ClinicalRegex.get_instrument_prefix_small()
        age_typo_prefix = ClinicalRegex.get_age_typo_prefix()
        word_digit_split_prefix = ClinicalRegex.get_word_digit_split_prefix()
        word_digit_mix_prefix = ClinicalRegex.get_word_digit_mix_prefix()
        word_digit_mix_prefix_small = ClinicalRegex.get_word_digit_mix_prefix_small()
        word_id_split_prefix = ClinicalRegex.get_word_id_split_prefix()
        word_section_prefix = ClinicalRegex.get_word_section_prefix()
        colon_prefix = ClinicalRegex.get_colon_prefix()
        temperature_prefix = ClinicalRegex.get_temperature_prefix(split_temperature)
        percentage_prefix = ClinicalRegex.get_percentage_prefix(split_percentage)
        value_range_1, value_range_2, value_range_3 = ClinicalRegex.get_value_range_prefixes()
        year_range_prefix = ClinicalRegex.get_year_range_prefix()
        short_digit_id_prefix = ClinicalRegex.get_short_digit_id_prefix()
        digit_symbol_word_prefix = ClinicalRegex.get_digit_symbol_word_prefix()
        digit_age_split_prefix = ClinicalRegex.get_digit_age_split_prefix()
        digit_word_short_prefix = ClinicalRegex.get_digit_word_short_prefix()
        digit_word_typo_prefix = ClinicalRegex.get_digit_word_typo_prefix()
        
        return [word_typo_prefix, word_symbol_digit_prefix, pager_prefix, age_word_prefix,\
                word_period_prefix, id_prefix, multiple_prefix, chemical_prefix, chemical_prefix_small,\
                instrument_prefix, instrument_prefix_small, age_typo_prefix, word_digit_split_prefix,\
                word_id_split_prefix, word_digit_mix_prefix, word_digit_mix_prefix_small, \
                word_section_prefix, colon_prefix, temperature_prefix,\
                percentage_prefix, value_range_1, value_range_2, value_range_3, year_range_prefix,\
                short_digit_id_prefix, digit_symbol_word_prefix, digit_age_split_prefix,\
                digit_word_short_prefix, digit_word_typo_prefix]
    
    @staticmethod
    def get_infixes():
        digit_infix = '(\d+(?P<sep>[-:]+)\d+((?P=sep)\d+)*)'
        return [digit_infix, ]
        