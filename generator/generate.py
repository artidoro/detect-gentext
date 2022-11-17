from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import more_itertools
import fire
import os
from tqdm import tqdm
import numpy as np

# local_rank = int(os.getenv('LOCAL_RANK', '0'))
# world_size = int(os.getenv('WORLD_SIZE', '1'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_prompts(text_list):
    return [" ".join(text.split()[:15]) for text in text_list]

def generate(
        pretrained_model_name_or_path="gpt2-md-covid",
        path_to_prompt_data="nela-covid-2020-train.json",
        path_to_generated_data="testoutputs.json",
        prompt_len=10,
        batch_size=40,
        min_length=20,
        max_length=None,
        do_sample=True,
        num_beams=1,
        temperature=1.0,
        top_k=500000,
        top_p=1.0,
        repetition_penalty=1.3, # 1.2 is good according to paper
        num_return_sequences=1,
        debug_doc_nb=None, # when specifying a number just generate for that doc
    ):
    """
    Data should be in the same format as the "run_clm.py" script expects. 
    JsonL file with fields including "text" with article content.

    batch_size is the number of documents to generate in parallel in a batch.
    """
    print("Loading model and tokenizer.")
    # Setting up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # To avoid generating @ that appear in NELA, we define a bad word list
    bad_words = ["@", "@ @", " @", "@ ", ".com", "cookies", "http", ".org"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids #add_prefix_space=True, 

    # Setting up model
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)
    print(f'Model on device {device}')
    print("Model loaded.")

    print("Loading texts.")
    # Load generated data (check what is already generated if any)
    data_already_generated = set()
    if os.path.exists(path_to_generated_data) and debug_doc_nb is None:
        with open(path_to_generated_data) as infile:
            data_already_generated = {json.loads(line.strip())['id'] for line in infile.readlines()}

    # Load text data (do not add already generated outputs)
    with open(path_to_prompt_data) as infile:
        data = []
        for line in infile.readlines():
            example = json.loads(line.strip())
            if example['id'] not in data_already_generated:
                data.append(example)

    if debug_doc_nb is not None:
        # data = [data[debug_doc_nb]]
        data = data[:debug_doc_nb]

    print("Starting generation.")
    if max_length is None:
        max_length = min(850, int(np.random.normal(550, 200)))
    for batch_data in tqdm(more_itertools.chunked(data, batch_size)):
        batch_prompts = extract_prompts([elt["text"] for elt in batch_data])
        prompt_tokens = tokenizer(
            batch_prompts,
            truncation=True,
            padding="max_length",
            return_tensors='pt',
            max_length=prompt_len,
        ).to(device)
        with torch.no_grad():
            # More parameters at https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_utils.GenerationMixin.generate
            generated_ids = model.generate(
                **prompt_tokens,
                bad_words_ids=bad_words_ids,
                min_length=min_length, # The minimum length of the sequence to be generated. defaults to 10
                max_length=max_length, # The maximum length of the sequence to be generated. defaults to model.config.max_length
                do_sample=do_sample, # Whether or not to use sampling ; use greedy decoding otherwise. default False
                num_beams=num_beams, # Number of beams for beam search. 1 means no beam search. defaults to 1
                temperature=temperature, # The value used to module the next token probabilities. defaults to 1.0
                top_k=top_k, # The number of highest probability vocabulary tokens to keep for top-k-filtering. defaults to 50
                top_p=top_p, # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                repetition_penalty=repetition_penalty, # The parameter for repetition penalty. 1.0 means no penalty  defaults to 1.0
                num_return_sequences=num_return_sequences, # The number of independently computed returned sequences for each element in the batch
            )
        # generated_texts = tokenizer.batch_decode(
        #     generated_ids, skip_special_tokens=True)

        generated_texts = tokenizer.batch_decode(
            generated_ids)

        if debug_doc_nb is None:
            for generated_text, prompt, example_dict in zip(generated_texts, batch_prompts, batch_data):
                example = {
                    **example_dict,
                    'prompt':prompt,
                    'generation':generated_text,
                }
                with open(path_to_generated_data, 'a') as outfile:
                    outfile.write(f'{json.dumps(example)}\n')
        else:
            print(tokenizer.batch_decode(prompt_tokens["input_ids"]))
            print(generated_texts)
            print(f'len {len(generated_ids[0])}')
            print(batch_data[0]["text"])

if __name__ == '__main__':
    fire.Fire(generate)