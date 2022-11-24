import torch
from fairseq.tokenizer import tokenize_line

BAR_START = "<0-0>"
BAR_END = "<0-255>"

POS_START = "<1-0>"
POS_END = "<1-127>"

INS_START = "<2-0>"
INS_END = "<2-127>"

PITCH_START = "<3-0>"
PITCH_END = "<3-255>"

DUR_START = "<4-0>"
DUR_END = "<4-127>"

VEL_START = "<5-0>"
VEL_END = "<5-31>"

SIG_START = "<6-0>"
SIG_END = "<6-253>"

TEMPO_START = "<7-0>"
TEMPO_END = "<7-48>"


# Functions to return range of indices for an Octuple element
def bar_range(label_dict): return label_dict.index(
    BAR_START), label_dict.index(BAR_END) + 1


def pos_range(label_dict): return label_dict.index(
    POS_START), label_dict.index(POS_END) + 1


def ins_range(label_dict): return label_dict.index(
    INS_START), label_dict.index(INS_END) + 1


def pitch_range(label_dict): return label_dict.index(
    PITCH_START), label_dict.index(PITCH_END) + 1


def dur_range(label_dict): return label_dict.index(
    DUR_START), label_dict.index(DUR_END) + 1


def vel_range(label_dict): return label_dict.index(
    VEL_START), label_dict.index(VEL_END) + 1


def sig_range(label_dict): return label_dict.index(
    SIG_START), label_dict.index(SIG_END) + 1


def tempo_range(label_dict): return label_dict.index(
    TEMPO_START), label_dict.index(TEMPO_END) + 1


def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
) -> torch.IntTensor:
    words = line_tokenizer(line)
    if reverse_order:
        words = list(reversed(words))
    nwords = len(words)
    ids = torch.IntTensor(nwords + 8 if append_eos else nwords)

    for i, word in enumerate(words):
        if add_if_not_exist:
            idx = self.add_symbol(word)
        else:
            idx = self.index(word)
        if consumer is not None:
            consumer(word, idx)
        ids[i] = idx
    if append_eos:
        ids[nwords:] = self.eos_index
    return ids