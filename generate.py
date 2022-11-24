import argparse
import os
import torch
from omegaconf import OmegaConf

from src.musicbert import MusicBERTModel
from src.processing.preprocess import encoding_to_MIDI
from src.processing.postprocess import postprocess

parser = argparse.ArgumentParser()


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


# Configuration file argument
parser.add_argument('--save_path', required=True, type=str, help='Specify the path to save the generated MIDI file.',
                    default='.')
parser.add_argument('--data', required=False, type=str,
                    help='The path to the preprocessed data.', default='')
parser.add_argument('--checkpoint', required=False, type=str,
                    help='The checkpoint file to load.', default='')
parser.add_argument('--config', required=False, type=str, help='Specify a configuration file, instead of using the '
                                                               'command-line optional arguments', default=None)
parser.add_argument('--sampling_method', required=False, type=str, choices=['seq', 'gibbs', 'fill_mask'],
                    help='The method to sample from the model.', default='seq')
parser.add_argument('--prompt', required=False, type=str,
                    help='Use a prompt to assist the generation.', default=None)
parser.add_argument('--n_tokens', required=False, type=int,
                    help='The number of tokens to unmask. Default is 20.', default=20)
parser.add_argument('--topk', required=False, type=int,
                    help='Include the top K probabilities from the output distribution. Default is 1.', default=1)
parser.add_argument('--topp', required=False, type=restricted_float,
                    metavar="[0.0-1.0]",
                    help='Specify a p-probability (0.0-1.0) to perform nucleus sampling. Default is 0.0.', default=0.0)
parser.add_argument('--temperature', required=False, type=restricted_float,
                    metavar="[0.0-1.0]",
                    help='Specify the temperature (0.0-1.0) to apply to the output probabilities. Default is 1.0.',
                    default=1.0)
parser.add_argument('--max_steps', required=False, type=int,
                    help='The number of steps for Gibbs sampling to run. Default is the length of the input sequence')

args = parser.parse_args()

if args.config is not None:
    cfg = OmegaConf.load(args.config)
    sampling_method = cfg['sampling_method']
    prompt_file = cfg['prompt']
    n_tokens = cfg['hyperparameters']['num_tokens']
    top_k = cfg['hyperparameters']['top_k']
    top_p = cfg['hyperparameters']['top_p']
    temperature = cfg['hyperparameters']['temperature']
    max_steps = cfg['hyperparameters']['max_steps']
    checkpoint = os.path.abspath(
        os.path.join(os.path.abspath(''), cfg['checkpoint']['path'], cfg['checkpoint']['file']))

    data_path = os.path.abspath(
        os.path.join(os.path.abspath(''), cfg['data']['path'], cfg['data']['bin_dir']))
else:
    sampling_method = args.sampling_method
    prompt_file = args.prompt
    n_tokens = args.n_tokens
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    max_steps = args.max_steps
    checkpoint = args.checkpoint
    data_path = args.data

print(sampling_method, prompt_file, n_tokens, top_k, top_p, temperature, max_steps, checkpoint, data_path)
user_dir = os.path.abspath(os.path.join(os.path.abspath(''), '', 'src/musicbert'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO]: Initializing model from {checkpoint}')
musicbert = MusicBERTModel.from_pretrained(
    model_name_or_path='src',
    checkpoint_file=checkpoint,
    data_name_or_path=data_path,
    # user_dir=user_dir
).to(device)

prompt = None
if prompt_file is not None:
    prompt = musicbert.prompt_from_midi(prompt_file, -1)  # -1 will use the whole midi
    print(f'[INFO]: Prompt created from {checkpoint}')
if __name__ == '__main__':
    print(f'[INFO]: Generating...')
    if sampling_method == 'seq':
        generated_seq, _ = musicbert.sequential_generation(
            prompt=prompt,
            n_tokens=n_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )

        print(generated_seq)
    elif sampling_method == 'gibbs':
        generated_seq = musicbert.gibbs_generation(
            max_steps=10,
            prompt=prompt,
            n_tokens=n_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)

        print(generated_seq)
    elif sampling_method == 'fill_mask':
        generated_seq, _ = musicbert.fill_mask(
            prompt=prompt,
            n_tokens=n_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
    e = postprocess(generated_seq, prompt)
    midi_obj = encoding_to_MIDI(e[8:-7])
    midi_obj.dump(os.path.join(args.save_path, 'generated_midi.mid'))
    print(f'[INFO]: Saved generated MIDI to {os.path.join(args.save_path, "generated_midi.mid")}')
