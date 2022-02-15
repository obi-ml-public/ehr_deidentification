from transformers import (
    BertConfig,
    BertForTokenClassification,
)

from .conditional_random_field_sub import ConditionalRandomFieldSub
from .crf_token_classifier_output import CRFTokenClassifierOutput


class CRFBertModelForTokenClassification(BertForTokenClassification):
    def __init__(
            self,
            config: BertConfig,
            crf_constraints
    ):
        super().__init__(config)
        self.crf = ConditionalRandomFieldSub(num_labels=config.num_labels, constraints=crf_constraints)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Or we use self.base_model - might work with auto model class
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        batch_size = logits.shape[0]
        sequence_length = logits.shape[1]
        loss = None
        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            loss = 0
            for seq_logits, seq_labels in zip(logits, labels):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_mask = seq_labels != -100
                seq_logits_crf = seq_logits[seq_mask].unsqueeze(0)
                seq_labels_crf = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(inputs=seq_logits_crf, tags=seq_labels_crf)
            loss /= batch_size
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return CRFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
