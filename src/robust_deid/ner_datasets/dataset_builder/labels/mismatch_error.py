# Exception thrown when there is a mismatch between a token and span
# The token and spans don't line up due to a tokenization issue
# E.g - 79M - span is AGE - 79, but token is 79M
# There is a mismatch and an error will be thrown - that is the token does
# not line up with the span
class MismatchError(Exception):
    pass
