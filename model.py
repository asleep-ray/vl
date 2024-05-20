import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import CLIPModel, T5ForConditionalGeneration, T5Tokenizer
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

FLAN_T5_BASE = "google/flan-t5-base"
CLIP_VIT_BASE_PATCH16 = "openai/clip-vit-base-patch16"


class IntegratedModel(nn.Module, GenerationMixin):
    def __init__(self):
        super(IntegratedModel, self).__init__()

        self.encoder = CLIPModel.from_pretrained(CLIP_VIT_BASE_PATCH16)

        flan_t5_model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_BASE)
        self.decoder = flan_t5_model.decoder
        self.lm_head = flan_t5_model.lm_head

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q", "v", "wi_0", "wi_1"], lora_dropout=0.05, bias="none"
        )
        self.decoder = get_peft_model(self.decoder, lora_config)

        self.tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_BASE)
        self.tokenizer.add_special_tokens({"bos_token": "<s>"})  # Add a custom start token if not already present

    def get_encoder_output(self, processed_image: torch.Tensor) -> torch.Tensor:
        """Get the encoded representation of the input image.

        Args:
            processed_image: The processed image tensor with shape of (bs, 3, 244, 244).

        Returns:
            encoder_outputs: The encoded representation of the input image with shape of (bs, 196, 768).
        """
        outputs = self.encoder.vision_model(processed_image)
        patch_embeddings = outputs.last_hidden_state  # shape (bs, 197, 768), patch embeddings are in the hidden_states
        patch_embeddings = patch_embeddings[:, 1:, :]  # shape (bs, 196, 768), Exclude the CLS token embedding
        encoder_outputs = patch_embeddings / patch_embeddings.norm(p=2, dim=-1, keepdim=True)  # Normalize embeddings
        return encoder_outputs

    def forward(self, processed_image, attention_mask=None, labels=None, **kwargs):
        """Forward pass of the model.

        Args:
            processed_image: Processed image tensor with shape (bs, 1, 3, 244, 244). TODO: Change to (bs, 3, 244, 244)
            attention_mask: The attention mask tensor.
            labels: The labels tensor with shape of (bs, seq_len).

        Returns:
            Seq2SeqLMOutput: The output of the model.
        """
        processed_image = processed_image.squeeze(1)  # shape (bs, 1, 3, 244, 244) -> (bs, 3, 244, 244)
        encoder_outputs = self.get_encoder_output(processed_image)  # shape (bs, 196, 768)

        decoder_input_ids = self._shift_right(labels)  # Shift the labels to the right for decoder input
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

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        return shifted_input_ids

    def generate(self, processed_image, max_new_tokens=50):
        """Generate text from the input image.

        Args:
            processed_image: The processed image tensor with shape of (bs, 3, 244, 244).
            max_new_tokens: The maximum number of tokens to generate.

        returns:
            The generated token ids.
        """
        encoder_outputs = self.get_encoder_output(processed_image)
        generated_ids = torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(0)  # Initialize with the start token

        for _ in range(max_new_tokens):
            decoder_outputs = self.decoder(input_ids=generated_ids, encoder_hidden_states=encoder_outputs)
            sequence_output = decoder_outputs.last_hidden_state
            next_token_logits = self.lm_head(sequence_output[:, -1, :])
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(-1)  # Greedily select the token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id == self.tokenizer.eos_token_id:  # Stop when end-of-sequence token is generated
                break

        return generated_ids
