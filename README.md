# Exploring Automatic Music Generation using Transformer encoder-based Language Models

You can find the full paper of the thesis [here](https://pergamos.lib.uoa.gr/uoa/dl/object/3245414)

**DISCLAIMER**: The codebase relies on the [MusicBERT](https://github.com/microsoft/muzic/tree/main/musicbert) open-source model.

## Installation
```shell
git clone https://github.com/aspil/bsc-thesis.git
cd bsc-thesis
./setup.sh
```
## 1. Environment
```
Python:        3.8
fairseq:       git+https://github.com/pytorch/fairseq@336942734c85791a90baa373c212d27e7c722662#egg=fairseq
```

The original MusicBERT checkpoints seem to work only with the version above.

## 2. Dataset
### 2.1 Preparing dataset
- The dataset used in the paper can be found in `data/raw` directory, but you can use any dataset.
- Run the dataset processing script. (`preprocess.py`)  
`python -u src/processing/preprocess.py`
- The script should prompt you to input the path of the midi zip and the path for the preprocessed output.  
    ```
    Dataset zip path: /data/raw/GiantMIDI-Baroque.zip
    OctupleMIDI output path: intermediate
    SUCCESS: test_midi.mid
    ```
- Binarize the raw text format dataset. (this script will read lmd_full_data_raw folder and output lmd_full_data_bin)
bash binarize_pretrain.sh GiantMIDI-Baroque  
`bash scripts/binarize_pretrain.sh data/baroque`

## 3. Train / Fine-tune
Both training and fine-tuning are done using the Masked Language Modelling task.
Therefore, for both scenarios use the following command:
```shell
bash scripts/train_mask.sh baroque base
```
In the current set, the data consists of Baroque musical pieces. The original pre-trained checkpoints can be
downloaded from [here](https://1drv.ms/u/s!Aq3YEPZCcV5ibz9ySjjNsEB74CQ). Create a `checkpoints` directory and place them there.

## 4. Generate

Use the following command to generate a musical piece:
```
python generate.py [-h] --save_path SAVE_PATH [--data DATA] [--checkpoint CHECKPOINT] [--config CONFIG] [--sampling_method {seq,gibbs,fill_mask}] [--prompt PROMPT] [--n_tokens N_TOKENS] [--topk TOPK] [--topp [0.0-1.0]] [--temperature [0.0-1.0]] [--max_steps MAX_STEPS]
```

The `config` parameter must be a YAML file, and if given any, all other arguments are ignored.