import torch
from dataclasses import dataclass
from transformers.modeling_outputs import TokenClassifierOutput


@dataclass
class CRFTokenClassifierOutput(TokenClassifierOutput):
    """
    The default TokenClassifierOutput returns logits, loss, hidden_states and attentions
    when we use the CRF module, we want the model.forward function to return the predicted
    sequence from the CRF module. So we introduce this class which subclasses TokenClassifierOutput
    and additionally returns the predictions tensor - which contains the sequences
    training examples.
    """
    predictions: torch.LongTensor = None
    scores: torch.LongTensor = None
