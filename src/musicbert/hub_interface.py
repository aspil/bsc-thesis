from types import MethodType
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from miditoolkit import MidiFile
from torch import Tensor

from src.processing.preprocess import encoding_to_MIDI, encoding_to_str, MIDI_to_encoding
from src.utils import *


class MusicBERTHubInterface(nn.Module):

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.task.source_dictionary.encode_line = MethodType(encode_line, self.task.source_dictionary)

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    def encode(self, sentence: str, *addl_sentences, append_eos=False, no_separator=False) -> torch.LongTensor:
        return self.task.source_dictionary.encode_line(line=sentence, append_eos=append_eos).long()

    def decode(self, sentence: torch.Tensor, return_list=False) -> str:
        sentence_list = sentence.tolist()
        seq = []
        inv_map = {v: k for k, v in self.task.source_dictionary.indices.items()}
        for token in sentence_list:
            seq.append(inv_map[token])

        if return_list:
            return seq
        else:
            return ' '.join(seq)

    @staticmethod
    def init_input(prompt: str, n_tokens: int) -> str:
        """
        Initialize the input to the model with either a sentence or a batch of sentences.

        Parameters
        ----------
        prompt : str
            The prompt to use for the model.
        n_tokens : int
            number of tokens to generate

        Returns
        -------
        List
            masked input
        """
        # If prompt is given, make sure the tokens are complete
        if prompt is not None:
            assert len(prompt.split()) % 8 == 0

        # Add BOS tokens and prompt if it is present

        if prompt is not None:
            input_tokens = prompt.split()
            if '<s>' not in prompt:
                input_tokens = ['<s>'] * 8 + input_tokens
        else:
            input_tokens = ['<s>'] * 8

        # Add mask tokens to the sequence to be generated
        input_tokens += ['<mask>'] * 8 * n_tokens

        return ' '.join(input_tokens)

    def override_tokens(self, inp, hard_values):
        for i, attrib in enumerate(hard_values):
            if attrib is not None:
                encoded_attrib = self.encode(sentence=attrib)
                assert encoded_attrib.size() == 1
                inp[i] = encoded_attrib[0]
        return inp

    def prompt_from_midi(self, midi_file, n_tokens=1):
        midi_obj = MidiFile(midi_file)
        encoded_midi = encoding_to_str(MIDI_to_encoding(midi_obj)).split()
        if n_tokens == -1 or n_tokens > len(encoded_midi) // 8:
            return ' '.join(encoded_midi[8:-7])
        return ' '.join(encoded_midi[8:(n_tokens + 1) * 8:])

    def sequential_generation(self, prompt=None, n_tokens=1, top_k: int = 0, top_p: float = 0.0, temperature=1.0):
        input_str = self.init_input(prompt, n_tokens)
        encoded_input = self.encode(sentence=input_str, append_eos=True)
        encoded_input.to(self._float_tensor.device)

        if prompt is not None:
            skip_prompt = len(prompt.split()) // 8
        else:
            skip_prompt = 0
        # generated_tokens = torch.tensor([])
        generated_tokens = []
        # Fill mask for one token at a time
        for i in range(skip_prompt, n_tokens + skip_prompt):

            # Get the subsequence up until the current first masked token and append the EOS token
            inp = torch.cat((encoded_input[:8 + (i + 1) * 8], encoded_input[-8:]), dim=0) \
                .view(1, -1).to(self._float_tensor.device)

            masked_index = (inp.squeeze() == self.task.mask_idx).nonzero(as_tuple=False)
            # TODO check if i can use simply the last 8 tokens from inp instead of masked_index

            if masked_index.dim() == 2:
                masked_index = masked_index.squeeze(1)
            # Pass the input to the model in order to fill the mask
            utils.model_eval(self.model)
            with torch.no_grad():
                features, _ = self.model.extract_features(inp)

            features = features.squeeze()  # no batch dimension in order to work with scatter in top_k_top_p_filtering
            logits = features[..., masked_index, :]

            logits = self.top_k_top_p_filtering(logits, top_k, top_p, temperature=temperature)

            probs = F.softmax(logits, dim=-1)
            if temperature != 1 or top_k != 0 or top_p != 0:
                # We sample from a multinomial distribution
                top_preds = torch.multinomial(probs, 1)
                top_preds = top_preds.reshape(-1)
            else:
                # We take the argmax or only choose the top candidate
                top_preds = torch.argmax(probs, dim=1)

            # check if the last 8 tokens are valid
            if self.is_valid_octuple(top_preds[-8:]):
                generated_tokens.append(top_preds[-8:])

        invalid_predictions = n_tokens - sum(preds.size(0) for preds in generated_tokens) // 8
        print(f"Skipped {invalid_predictions} invalid prediction"
              + ("s" if invalid_predictions > 1 else ""))

        return (prompt or '') + ' ' + ' '.join([self.decode(token) for token in generated_tokens]), invalid_predictions

    def fill_mask(self, prompt=None, n_tokens=1, top_k: int = 0, top_p: float = 0.0, temperature=1,
                  filter_invalid=False):

        masked_input = self.init_input(prompt, n_tokens)

        tokens = self.task.source_dictionary.encode_line(
            line=masked_input,
            append_eos=True,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero(as_tuple=False)

        if masked_index.dim() == 2:
            masked_index = masked_index.squeeze(1)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        utils.model_eval(self.model)
        with torch.no_grad():
            features, _ = self.model.extract_features(
                tokens.long().to(self._float_tensor.device))

        features = features.squeeze()  # no batch dimension in order to work with scatter in top_k_top_p_filtering
        logits = features[..., masked_index, :]
        logits = self.top_k_top_p_filtering(logits, top_k, top_p, temperature=temperature)

        probs = F.softmax(logits, dim=-1)

        if temperature != 1 or top_k != 0 or top_p != 0:
            # We sample from a multinomial distribution
            top_preds = torch.multinomial(probs, 1)
            top_preds = top_preds.reshape(-1)
        else:
            # We take the argmax or only choose the top candidate
            top_preds = torch.argmax(probs, dim=1)

        if filter_invalid:
            generated_tokens = []
            for i in range(0, len(top_preds), 8):
                if self.is_valid_octuple(top_preds[i:i + 8]):
                    generated_tokens.append(top_preds[i: i + 8])

            invalid_predictions = n_tokens - sum(preds.size(0) for preds in generated_tokens) // 8

            print(f"Skipped {invalid_predictions} invalid prediction"
                  + ("" if invalid_predictions == 1 else "s"))

        return (prompt or '') + ' ' + ' '.join([self.decode(token) for token in generated_tokens]), invalid_predictions

    def gibbs_generation(self, max_steps=50, prompt=None, n_tokens=1, all_masks=True, top_k: int = 0,
                         top_p: float = 0.0,
                         temperature=1.0):
        if n_tokens > 0 and prompt is None:
            if all_masks:  # case 1: all masks
                input_str = self.init_input(None, n_tokens)

            else:  # case 2: random initial sample
                input_str = self.fill_mask(None, n_tokens, top_k=0, top_p=0, temperature=1, filter_invalid=False)

        elif n_tokens > 0 and prompt is not None:  # case 3: generate n_tokens randomly with prompt context
            input_str = self.init_input(prompt, n_tokens)

        elif n_tokens == -1 and prompt is not None:  # case 4: replace tokens taken from a MIDI file
            input_str = '<s> ' * 8 + prompt  # TODO merge with first if by tweaking init_input

        else:
            raise ValueError("Invalid combination of prompt and n_tokens parameters")

        encoded_input = self.encode(sentence=input_str, append_eos=True)

        encoded_input.to(self._float_tensor.device)
        output = encoded_input.clone()

        random_idx_src = encoded_input.squeeze()
        for i in range(max_steps):
            rand_idx = torch.randint(0, random_idx_src.size(0) - 7, (1,)).item()

            sample_index = torch.arange(rand_idx, rand_idx + 8)

            encoded_input[sample_index] = self.task.mask_idx
            utils.model_eval(self.model)
            with torch.no_grad():
                features, _ = self.model.extract_features(
                    encoded_input.unsqueeze(0).long().to(self._float_tensor.device))

            features = features.squeeze()

            logits = features[..., sample_index, :]
            logits = self.top_k_top_p_filtering(logits, top_k, top_p, temperature=temperature)

            probs = F.softmax(logits, dim=-1)
            if temperature != 1 or top_k != 0 or top_p != 0:
                # We sample from a multinomial distribution
                top_preds = torch.multinomial(probs, 1)
                top_preds = top_preds.reshape(-1)
            else:
                # We take the argmax or only choose the top candidate
                top_preds = torch.argmax(probs, dim=1)

            sorted_top_preds, _ = torch.sort(top_preds)
            if self.is_valid_octuple(sorted_top_preds):
                mask = torch.ones_like(random_idx_src, dtype=bool)
                mask[sample_index] = False
                random_idx_src = random_idx_src[mask]
                output[sample_index] = sorted_top_preds

        return self.decode(output)

    def is_valid_octuple(self, octuple):
        return octuple[0] in list(range(*bar_range(self.task.source_dictionary))) \
               and octuple[1] in list(range(*pos_range(self.task.source_dictionary))) \
               and octuple[2] in list(range(*ins_range(self.task.source_dictionary))) \
               and octuple[3] in list(range(*pitch_range(self.task.source_dictionary))) \
               and octuple[4] in list(range(*dur_range(self.task.source_dictionary))) \
               and octuple[5] in list(range(*vel_range(self.task.source_dictionary))) \
               and octuple[6] in list(range(*sig_range(self.task.source_dictionary))) \
               and octuple[7] in list(range(*tempo_range(self.task.source_dictionary))) \
               and self.task.source_dictionary.bos_index not in octuple \
               and self.task.source_dictionary.eos_index not in octuple

    def top_k_top_p_filtering(
            self,
            logits: Tensor,
            top_k: int = 0,
            top_p: float = 1.0,
            filter_value: float = -float("Inf"),
            min_tokens_to_keep: int = 1,
            temperature: float = 1.0,
    ) -> Tensor:
        logits.div_(temperature)
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p

            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

            logits[indices_to_remove] = filter_value
        return logits

    # save generation to midi file
    def save_midi(self, generated_sequence, filename):
        midi_obj = encoding_to_MIDI(generated_sequence)
        if '.mid' not in filename or '.midi' not in filename:
            filename += '.mid'
        midi_obj.dump(filename)
