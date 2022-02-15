import torch
from typing import List, Tuple, NoReturn

from allennlp.modules import ConditionalRandomField


class ConditionalRandomFieldSub(ConditionalRandomField):
    """
    Implement a CRF layer
    The code is borrowed from allennlp, We could have used it directly but we had
    to subclass since using the code directly was throwing an error saying the mask
    tensor could not be found on the GPU. So we subclass and it put the mask tensor
    on the right device. Refer to allennlp for more details
    """

    def __init__(self, num_labels: int, constraints: List[Tuple[int, int]]) -> NoReturn:
        """
        Initialize the allennlp class with the number of labels and constraints
        Args:
            num_labels (int): The number of possible tags/labels (B-AGE, I-DATE, etc)
            constraints (List[Tuple[int, int]): Are there any constraints for certain tag transitions. For example
                                                dont allow transitions from B-DATE to I-MRN etc
        """
        super().__init__(num_labels, constraints)

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Computes the log likelihood.
        The only change we make is moving the mask tensor to the same device as the inputs
        Args:
            inputs (torch.Tensor): Model logits
            tags (torch.Tensor): True labels
            mask (torch.BoolTensor): Mask
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = mask.to(torch.bool, device=inputs.device)
        # Compute the CRF loss
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        # Return crf loss
        return torch.sum(log_numerator - log_denominator)
