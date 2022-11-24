from src.processing.preprocess import e2t, str_to_encoding, t2e
from src.utils import bar_range, pos_range, ins_range, pitch_range, dur_range, vel_range, sig_range, tempo_range


def valid_token(token, musicbert_dictionary):
    pred = token.split()
    return pred[0] not in list(range(*bar_range(musicbert_dictionary))) \
        and pred[1] not in list(range(*pos_range(musicbert_dictionary))) \
        and pred[2] not in list(range(*ins_range(musicbert_dictionary))) \
        and pred[3] not in list(range(*pitch_range(musicbert_dictionary))) \
        and pred[4] not in list(range(*dur_range(musicbert_dictionary))) \
        and pred[5] not in list(range(*vel_range(musicbert_dictionary))) \
        and pred[6] not in list(range(*sig_range(musicbert_dictionary))) \
        and pred[7] not in list(range(*tempo_range(musicbert_dictionary))) \
        and "<s>" not in pred \
        and "</s>" not in pred


# def expand_notes(x, notes_per_bar, instrument, ):
def enforce_instrument(x, instrument):
    for token in x:
        pred = token.split()
        if pred[2] == instrument:
            return token
    return None


def token_to_encoding(t):
    return int(t[3: -1])


def postprocess(generated_sequence, prompt=None, filter_instrument=True, notes_per_bar=4):
    # TODO - correct bar and position of generated tokens, look into duration and dur_enc logic
    e = str_to_encoding(generated_sequence)
    e = [list(t) for t in e]

    if filter_instrument:
        for i in range(0, len(e)):
            e[i][2] = token_to_encoding('<2-0>')

    return [tuple(i) for i in e]
