# Threat Scenarios and Best Practices for Neural Fake News Detection
We present the resources associated with our paper published at COLING 2022. 

> https://aclanthology.org/2022.coling-1.106/

This repo contains code to
1. Finetune a generator to a new domain (COVID here)
2. Sample text from a generator
3. Train a detector on generated text

We also provide checkpoints and preprocessed outputs to directly train detectors or test them.


## Implementation
### Setup
Our code uses [PyTorch](https://pytorch.org/), [Huggingface Transformers Library](https://github.com/huggingface/transformers/), and [Simpletransformer](https://github.com/ThilinaRajapakse/simpletransformers/). You can install the requirements using pip:
```
pip install -r requirements.txt
```

### Generator Finetuning to COVID Domain
We finetune the generators to the COVID domain by adapting the language-modeling example from the [Huggingface Transformers Library](https://github.com/huggingface/transformers/). You can find the finetuning code in the `./generation/` directory with example scripts to finetune generators at `./generation/scripts`. In some of our experiments we used Deepspeed to finetune larger models. Deepspeed lets you partition the optimizer, gradient, and parameters of the model (see more details [here](https://huggingface.co/docs/transformers/main_classes/deepspeed)).

If you want to finetune the NELA data, you will have to download it from the [original repository](https://github.com/MELALab/nela-gt#limitations), process it using the code found at `./data/prepare_nela.py.`, and use it with our code to finetune a new generator.

We also provide checkpoints of the finetuned generators in our experiments at the following [Drive link](https://drive.google.com/drive/folders/1E3YLoIWxBgAtvq2mu5CDugEVByLxZP9M?usp=share_link).

These are the available finetuned generators:
- gpt2-sm-covid  
- gpt2-lg-covid  
- gpt2-md-covid  
- gpt2-xl-covid  
- gptneo-covid

### Text Generation
We provide our script to generate text using a prompt with huggingface checkpoints. Note that the NELA dataset is anonymized with "@" signs (see [here](https://github.com/MELALab/nela-gt#limitations) for more details) so to generate data that did not contain the "@" sign, we ignore it during the sampling process.

The text generation code is found in the `./generation/` directory.

We provide the generated outputs from our finetuned models at the following [Drive link](https://drive.google.com/drive/folders/1E3YLoIWxBgAtvq2mu5CDugEVByLxZP9M?usp=share_link). Use the "processed" version which is anonymized like NELA to train detectors.


### Detector Training
You can find the detector training code in the `detector` directory with example scripts to train the detector at `./detector/scripts`.

We provide the checkpoints for detectors discussed in the paper at the following [Drive link](https://drive.google.com/drive/folders/1E3YLoIWxBgAtvq2mu5CDugEVByLxZP9M?usp=share_link). All the checkpoints should have an `eval_results.txt` file with the performance of the detector in domain (on the same generator the detector was trained on).
The checkpoints available are:

| model name | base model | name generator on which detector was trained on | decoding strategy |
| --- |---|---|---|
albert-base-v2_large-762M-nucleus |                   albert-base-v2 |              GPT-2 (OpenAI outputs) large-762M-nucleus | nucleus   |
albert-base-v2_medium-345M-nucleus |                  albert-base-v2 |              GPT-2 (OpenAI outputs) medium-345M-nucleus | nucleus  |
albert-base-v2_small-117M-nucleus |                   albert-base-v2 |              GPT-2 (OpenAI outputs) small-117M-nucleus | nucleus  |
albert-base-v2_xl-1542M |                             albert-base-v2 |              GPT-2 (OpenAI outputs) xl-1542M | random |
albert-base-v2_xl-1542M-k40 |                         albert-base-v2 |              GPT-2 (OpenAI outputs) xl-1542M-k40 | top-k=40 |
albert-base-v2_xl-1542M-nucleus |                     albert-base-v2 |              GPT-2 (OpenAI outputs) xl-1542M-nucleus | nucleus |
bert-base-cased_large-762M-nucleus |                  bert-base-cased |             GPT-2 (OpenAI outputs) large-762M-nucleus | nucleus |
bert-base-cased_medium-345M-nucleus |                 bert-base-cased |             GPT-2 (OpenAI outputs) medium-345M-nucleus | nucleus |
bert-base-cased_small-117M-nucleus |                  bert-base-cased |             GPT-2 (OpenAI outputs) small-117M-nucleus | nucleus |
bert-base-cased_xl-1542M |                            bert-base-cased |             GPT-2 (OpenAI outputs) xl-1542M | random |
bert-base-cased_xl-1542M-k40 |                        bert-base-cased |             GPT-2 (OpenAI outputs) xl-1542M-k40 | top-k=40 |
bert-base-cased_xl-1542M-nucleus |                    bert-base-cased |             GPT-2 (OpenAI outputs) xl-1542M-nucleus | nucleus |
bert-large-cased_large-762M-nucleus |                 bert-large-cased |            GPT-2 (OpenAI outputs) large-762M-nucleus | nucleus |
bert-large-cased_medium-345M-nucleus |                bert-large-cased |            GPT-2 (OpenAI outputs) medium-345M-nucleus | nucleus |
bert-large-cased_small-117M-nucleus |                 bert-large-cased |            GPT-2 (OpenAI outputs) small-117M-nucleus | nucleus |
bert-large-cased_xl-1542M |                           bert-large-cased |            GPT-2 (OpenAI outputs) xl-1542M | random |
bert-large-cased_xl-1542M-k40 |                       bert-large-cased |            GPT-2 (OpenAI outputs) xl-1542M-k40 | top-k=40 |
bert-large-cased_xl-1542M-nucleus |                   bert-large-cased |            GPT-2 (OpenAI outputs) xl-1542M-nucleus | nucleus |
electra-large-discriminator_large-762M-nucleus |      electra-large-discriminator | GPT-2 (OpenAI outputs) large-762M-nucleus | nucleus |
electra-large-discriminator_medium-345M-nucleus |     electra-large-discriminator | GPT-2 (OpenAI outputs) medium-345M-nucleus | nucleus |
electra-large-discriminator_xl-1542M |                electra-large-discriminator | GPT-2 (OpenAI outputs) xl-1542M | random |
electra-large-discriminator_xl-1542M-k40 |            electra-large-discriminator | GPT-2 (OpenAI outputs) 1542M-k40 | top-k=40 |
electra-large-discriminator_xl-1542M-nucleus |        electra-large-discriminator | GPT-2 (OpenAI outputs) 1542M-nucleus | nucleus |
electra-small-discriminator_large-762M-nucleus |      electra-small-discriminator | GPT-2 (OpenAI outputs) 762M-nucleus | nucleus |
electra-small-discriminator_medium-345M-nucleus |     electra-small-discriminator | GPT-2 (OpenAI outputs) 345M-nucleus | nucleus |
electra-small-discriminator_small-117M-nucleus |      electra-small-discriminator | GPT-2 (OpenAI outputs) small-117M-nucleus | nucleus |
electra-small-discriminator_xl-1542M |                electra-small-discriminator | GPT-2 (OpenAI outputs) xl-1542M | random |
electra-small-discriminator_xl-1542M-k40 |            electra-small-discriminator | GPT-2 (OpenAI outputs) 1542M-k40 | top-k=40 |
electra-small-discriminator_xl-1542M-nucleus |        electra-small-discriminator | GPT-2 (OpenAI outputs) 1542M-nucleus | nucleus |
roberta-base_large-762M-nucleus |                     roberta-base |                GPT-2 (OpenAI outputs) large-762M-nucleus | nucleus |
roberta-base_medium-345M-nucleus |                    roberta-base |                GPT-2 (OpenAI outputs) medium-345M-nucleus | nucleus |
roberta-base_small-117M-nucleus |                     roberta-base |                GPT-2 (OpenAI outputs) small-117M-nucleus | nucleus |
roberta-base_xl-1542M |                               roberta-base |                GPT-2 (OpenAI outputs) xl-1542M | random |
roberta-base_xl-1542M-k40 |                           roberta-base |                GPT-2 (OpenAI outputs) xl-1542M-k40 | top-k=40 |
roberta-base_xl-1542M-nucleus |                       roberta-base |                GPT-2 (OpenAI outputs) xl-1542M-nucleus | nucleus |
roberta-large_large-762M-nucleus |                    roberta-large |               GPT-2 (OpenAI outputs) large-762M-nucleus | nucleus |
roberta-large_medium-345M-nucleus |                   roberta-large |               GPT-2 (OpenAI outputs) medium-345M-nucleus | nucleus |
roberta-large_small-117M-nucleus |                    roberta-large |               GPT-2 (OpenAI outputs) small-117M-nucleus | nucleus |
roberta-large_xl-1542M |                              roberta-large |               GPT-2 (OpenAI outputs) xl-1542M | random |
roberta-large_xl-1542M-k40 |                          roberta-large |               GPT-2 (OpenAI outputs) xl-1542M-k40 | top-k=40 |
roberta-large_xl-1542M-nucleus |                      roberta-large |               GPT-2 (OpenAI outputs) xl-1542M-nucleus | nucleus |
roberta-large-gpt2-sm-covid-random |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-sm-covid-random |  random |
roberta-large-gpt2-sm-covid-topk40 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-sm-covid-topk40 | top-k=40 |
roberta-large-gpt2-sm-covid-topp96 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-sm-covid-topp96 | nucleus top p=0.96 |
roberta-large-gpt2-lg-covid-random |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-lg-covid-random | random |
roberta-large-gpt2-lg-covid-topk40 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-lg-covid-topk40 | top k=40 |
roberta-large-gpt2-lg-covid-topp96 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-lg-covid-topp96 | nucleus top p=0.96 |
roberta-large-gpt2-md-covid-random |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-md-covid-random | random |
roberta-large-gpt2-md-covid-topk40 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-md-covid-topk40 | top k=40 |
roberta-large-gpt2-md-covid-topp96 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-md-covid-topp96 | nucleus top p=0.96 |
roberta-large-gpt2-xl-covid-random |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-xl-covid-random |  random |
roberta-large-gpt2-xl-covid-topk40 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-xl-covid-topk40 | top k=40 |
roberta-large-gpt2-xl-covid-topp96 |                  roberta-large |               GPT-2 (Covid finetuned generators) gpt2-xl-covid-topp96 | nucleus top p=0.96 |
roberta-large-gptneo-covid-random |                   roberta-large |               GPT-2 (Covid finetuned generators) gptneo-covid-random | random |
roberta-large-gptneo-covid-topk40 |                   roberta-large |               GPT-2 (Covid finetuned generators) gptneo-covid-topk40 | top k=40 |
roberta-large-gptneo-covid-topp96 |                   roberta-large |               GPT-2 (Covid finetuned generators) gptneo-covid-topp96 | nucleus top p=0.96 |
roberta-large-gpt3-davinci-002-webtext |              roberta-large |               GPT-3 (Our generations) davinci-002-webtext | nucleus top p=0.96 |


 
## Data
### OpenAI Data
The GPT-2 outputs and WebText data can be obtained by following [these instructions](https://github.com/openai/gpt-2-output-dataset).

### GPT-3 Data
We provide GPT-3 generations described in the paper (using Davinci-002) sampled with WebText prompts at this [Drive link](https://drive.google.com/drive/folders/1E3YLoIWxBgAtvq2mu5CDugEVByLxZP9M?usp=share_link). The code to reproduce the generation using GPT-3 is at `./generator/generate_gpt3.py`. 

### COVID Human Written Data
To replicate our experiments on COVID, you will need to dowload the [NELA 2020](https://github.com/MELALab/nela-gt#limitations) data and process it using the code found at `./data/prepare_nela.py`. 

### COVID Generated Data
We provide the generated outputs from our finetuned models at the following [Drive link](https://drive.google.com/drive/folders/1E3YLoIWxBgAtvq2mu5CDugEVByLxZP9M?usp=share_link). For the covid data, you will have to process the generated data to match the anonymized data from NELA using the code in `./data/prepare_generated_data_for_classification.py`.

We provide the preprocessed (anonymized) data at this [Drive link](https://drive.google.com/drive/folders/1E3YLoIWxBgAtvq2mu5CDugEVByLxZP9M?usp=share_link):

| generator | sampling |
|---|---|
| gpt2-lg-covid | random |
| gpt2-lg-covid | topk40 |
| gpt2-lg-covid | topp96 |
| gpt2-md-covid | random |
| gpt2-md-covid | topk40 |
| gpt2-md-covid | topp96 |
| gpt2-sm-covid | random |
| gpt2-sm-covid | topk40 |
| gpt2-sm-covid | topp96 |
| gpt2-xl-covid | random |
| gpt2-xl-covid | topk40 |
| gpt2-xl-covid | topp96 |
| gptneo-covid | random |
| gptneo-covid | topk40 |
| gptneo-covid | topp96 |

## Disclaimers

Given the large number of experiments, we did not manage to perform full hyperparameter tuning for all the models. Generators were finetuned using recommended hyperparameters on Huggingface. And following hyperparameter search on Roberta-large, we used the same hyperparameters for all the detectors.

## Contact

Feel free to reach out to my email or open a Github issue if you run into problems.

## Citation
```
@inproceedings{pagnoni-etal-2022-threat,
    title = "Threat Scenarios and Best Practices to Detect Neural Fake News",
    author = "Pagnoni, Artidoro  and
      Graciarena, Martin  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.106",
    pages = "1233--1249",
    abstract = "In this work, we discuss different threat scenarios from neural fake news generated by state-of-the-art language models. Through our experiments, we assess the performance of generated text detection systems under these threat scenarios. For each scenario, we also identify the minimax strategy for the detector that minimizes its worst-case performance. This constitutes a set of best practices that practitioners can rely on. In our analysis, we find that detectors are prone to shortcut learning (lack of out-of-distribution generalization) and discuss approaches to mitigate this problem and improve detectors more broadly. Finally, we argue that strong detectors should be released along with new generators.",
}
```
