# Get the number of violation in the predicted output
from typing import NoReturn, Sequence, Tuple


class Violations(object):
    """
    This class is used to compute the violations in the predictions
    A violation is something like i.e., how many times `I-TYPE` follows `O` 
    or a tag of a different type.
    """

    @staticmethod
    def get_prefixes(notation: str) -> Tuple[str, str, str, str]:
        """
        Initialize variables that are used to check for violations based on the notation
        Args:
            notation (str): The NER labelling scheme
        Returns:
            prefix_single, prefix_begin, prefix_inside, prefix_end, prefix_outside (Tuple[str, str, str, str]): The prefixes in
                                                                                                                the labels based 
                                                                                                                on the labelling 
                                                                                                                scheme
        """
        # Define the variables that represent the tags based on the notation
        if notation == 'BIO':
            prefix_single = 'B'
            prefix_begin = 'B'
            prefix_inside = 'I'
            prefix_end = 'I'
            prefix_outside = 'O'
        elif notation == 'BIOES':
            prefix_single = 'S'
            prefix_begin = 'B'
            prefix_inside = 'I'
            prefix_end = 'E'
            prefix_outside = 'O'
        elif notation == 'BILOU':
            prefix_single = 'U'
            prefix_begin = 'B'
            prefix_inside = 'I'
            prefix_end = 'L'
            prefix_outside = 'O'
        elif notation == 'IO':
            prefix_single = 'I'
            prefix_begin = 'I'
            prefix_inside = 'I'
            prefix_end = 'I'
            prefix_outside = 'O'
        else:
            raise ValueError('Invalid Notation')
        return prefix_single, prefix_begin, prefix_inside, prefix_end, prefix_outside

    @staticmethod
    def get_violations(tag_sequence: Sequence[str], notation: str) -> int:
        """
        Compute the violations in the predictions/labels
        A violation is something like i.e., how many times `I-TYPE` follows `O` 
        or a tag of a different type.
        Args:
            tag_sequence (Sequence[str]): The predictions/labels (e.g O, B-DATE, I-AGE) 
            notation (str): The NER labelling scheme
        Returns:
            count (int): The number of violations
        """
        prefix_single, prefix_begin, prefix_inside, prefix_end, prefix_outside = Violations.get_prefixes(
            notation=notation)
        count = 0
        start_tag = None
        prev_tag_type = prefix_single
        for tag in tag_sequence:
            tag_split = tag.split('-')
            # Check if the current tag is the beginning of a span or is a unit span (span of 1 token)
            if tag_split[0] in [prefix_begin, prefix_single]:
                # If the previous tag is not O, END (E,L) or UNIT (S, U) then it is a violation
                # Since this span started and the previous span did not end
                if prev_tag_type not in [prefix_outside, prefix_end, prefix_single]:
                    count += 1
                start_tag = tag_split[1]
                prev_tag_type = tag_split[0]
            # Check if the current tag is the inside/end of a span
            # If it is preceeded by the O tag - then it is a violation - because this span
            # does not have a begin tag (B)
            elif tag_split[0] in [prefix_inside, prefix_end] and prev_tag_type == prefix_outside:
                count += 1
                start_tag = tag_split[1]
                prev_tag_type = tag_split[0]
            # Check if the current tag is the inside/end of a span - if the type of the span
            # is different then it is a violation. E.g DATE followed by AGE when the DATE tag has not ended
            elif tag_split[0] in [prefix_inside, prefix_end] and prev_tag_type != prefix_outside:
                if prev_tag_type not in [prefix_inside, prefix_begin]:
                    count += 1
                elif tag_split[1] != start_tag:
                    count += 1
                start_tag = tag_split[1]
                prev_tag_type = tag_split[0]
            else:
                start_tag = None
                prev_tag_type = prefix_outside
        return count
