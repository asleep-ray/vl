import torch

# from peft import LoraConfig, get_peft_model
# from PIL import Image
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import CLIPModel, T5ForConditionalGeneration, T5Tokenizer
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput


class IntegratedModel(nn.Module, GenerationMixin):
    def __init__(self, model_name="google/flan-t5-base"):
        super(IntegratedModel, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # Add a custom start token if not already present
        start_token = "<s>"
        self.tokenizer.get_vocab()
        if start_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"bos_token": start_token})

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.decoder = self.model.decoder
        self.lm_head = self.model.lm_head
        self.config = self.model.config

        self.encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, processed_image, attention_mask=None, labels=None, **kwargs):
        has_value = any(value is not None for value in kwargs.values())
        if has_value:
            raise ValueError("This model does not support extra kwargs.")

        processed_image = processed_image.squeeze(1)
        # Encode the input text
        outputs = self.encoder.vision_model(processed_image)

        # The patch embeddings are located in the hidden_states of the model output
        patch_embeddings = outputs.last_hidden_state

        # Exclude the CLS token embedding
        patch_embeddings = patch_embeddings[:, 1:, :]

        # Normalize the embeddings
        encoder_outputs = patch_embeddings / patch_embeddings.norm(p=2, dim=-1, keepdim=True)

        # Shift the labels to the right for decoder input
        decoder_input_ids = self._shift_right(labels)

        # Decode the encoded representation
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=encoder_outputs, encoder_attention_mask=attention_mask
        )

        # Get the sequence output from the decoder
        sequence_output = decoder_outputs.last_hidden_state  # = decoder_outputs[0]

        # Apply the language model head to get the final logits
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            labels = labels.to(lm_logits.device)  # move labels to correct device to enable PP
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.tokenizer.bos_token_id
        pad_token_id = 0

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def generate(self, processed_image, max_new_tokens=50):

        # Encode the image
        outputs = self.encoder.vision_model(processed_image)

        # The patch embeddings are located in the hidden_states of the model output
        patch_embeddings = outputs.last_hidden_state

        # Exclude the CLS token embedding
        patch_embeddings = patch_embeddings[:, 1:, :]

        # Normalize the embeddings
        encoder_outputs = patch_embeddings / patch_embeddings.norm(p=2, dim=-1, keepdim=True)

        # Initialize decoder input IDs with the start token
        decoder_input_ids = torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(0)

        # Greedily generate text
        generated_ids = decoder_input_ids

        for _ in range(max_new_tokens):
            # Decode the encoded representation
            decoder_outputs = self.decoder(input_ids=generated_ids, encoder_hidden_states=encoder_outputs)

            # Get the logits for the next token
            sequence_output = decoder_outputs.last_hidden_state
            next_token_logits = self.lm_head(sequence_output[:, -1, :])

            # Greedily select the token with the highest probability
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(-1)

            # Append the predicted token ID to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # If the end-of-sequence token is generated, stop
            if next_token_id == self.tokenizer.eos_token_id:
                break

        return generated_ids, encoder_outputs
