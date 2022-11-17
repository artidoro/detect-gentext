import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import more_itertools
import fire
import deepspeed
import os
from tqdm import tqdm
import numpy as np
import time


# local_rank = int(os.getenv('LOCAL_RANK', '0'))
# world_size = int(os.getenv('WORLD_SIZE', '1'))
import openai
openai.api_key = ""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_prompts(text_list, prompt_len=15):
    return [extract_prompt(text, prompt_len) for text in text_list]

def extract_prompt(text, prompt_len=15):
    return " ".join(text.split()[:prompt_len])

def generate(
        engine='text-davinci-002',  #text-davinci-002     Model prices at: https://openai.com/api/pricing/
        path_to_prompt_data="xl-1542M-nucleus.train.jsonl",
        path_to_generated_data="gpt-3.json",
        prompt_len=10,
        max_length=None,
        temperature=1.0,
        top_p=0.96,
        repetition_penalty=0.5, # 1.2 is good according to paper
        debug_doc_nb=None, # when specifying a number just generate for that doc
        batch_size=16,
        max_num_docs=32,
    ):
    """
    Data should be in the same format as the "run_clm.py" script expects. 
    JsonL file with fields including "text" with article content.

    batch_size is the number of documents to generate in parallel in a batch.
    """ 

    print("Loading texts.")
    # Load generated data (check what is already generated if any)
    data_already_generated = set()
    if os.path.exists(path_to_generated_data) and debug_doc_nb is None:
        with open(path_to_generated_data) as infile:
            data_already_generated = {json.loads(line.strip())['id'] for line in infile.readlines()}

    # Load text data (do not add already generated outputs)
    with open(path_to_prompt_data) as infile:
        data = []
        c = 0
        for line in infile.readlines():
            example = json.loads(line.strip())
            if example['id'] not in data_already_generated or debug_doc_nb is not None:
                data.append(example)
            else:
                c +=1
        print(f'WARNING: {c} already generated. Will generated the rest.')


    if max_num_docs is not None:
        print(f'WARNING: Truncating dataset to only contain {max_num_docs} examples.')
        data = data[:max_num_docs]

    if debug_doc_nb is not None:
        # data = [data[debug_doc_nb]]
        data = data[:debug_doc_nb]
    
    print("Starting generation.")

    for batch_data in tqdm(more_itertools.chunked(data, batch_size), total=len(data)/batch_size):
        batch_prompts = extract_prompts([elt["text"] for elt in batch_data])
        if max_length is None:
            max_length = min(850, int(np.random.normal(500, 200)))
        # prompt = extract_prompt(example_dict['text'], prompt_len)
        
        gpt3_api_output = None
        while gpt3_api_output is None:
            try:
                gpt3_api_output = openai.Completion.create(
                    engine=engine,
                    prompt=batch_prompts,
                    temperature=temperature,
                    max_tokens=max_length,
                    top_p=top_p,
                    #   logprobs=1,
                    stream=False,
                    echo=True,
                    frequency_penalty=repetition_penalty,
                    #   presence_penalty=0,
                    )
            except:
                sleep_time = 10
                print(f'Exception. Trying again in {sleep_time}s.')
                time.sleep(10)
                continue


            

        if debug_doc_nb is None:
            for single_output, example_dict, prompt in zip(gpt3_api_output['choices'], batch_data, batch_prompts):
                example = {
                    **example_dict,
                    'prompt': prompt,
                    'generation': single_output['text'],
                    'gpt3-api-output': single_output,
                }
                with open(path_to_generated_data, 'a') as outfile:
                    outfile.write(f'{json.dumps(example)}\n')
        else:
            print(batch_prompts[0])
            print(gpt3_api_output['choices'][0]['text'])
            print(f"len {len(gpt3_api_output['choices'][0]['text'].split())}")
            print(batch_data[0]["text"])

if __name__ == '__main__':
    fire.Fire(generate)