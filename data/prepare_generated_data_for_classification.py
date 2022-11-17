import json
import fire
import random
from tqdm import tqdm
import os

from nltk.tokenize import word_tokenize
tokenizer = word_tokenize

def apply_at_masking_and_filter(text, at_masking=True):
    new_lines = text.split('\n')
    tokenized_new_lines = []
    num_tokens = 0
    for line in new_lines:
        tokens = tokenizer(line)
        tokenized_new_lines.append(tokens)
        num_tokens += len(tokens)

    new_lines_filtered = []
    id_count = 0
    for tokens_line in tokenized_new_lines:
        found_long_token = False
        tokens_new = []
        for tok in tokens_line:
            if len(tok) > 35:
                found_long_token = True
                break
            if (
                    (
                        (id_count % 100 < 7 and num_tokens >= 200 and id_count > 7) or
                        (id_count % 20 < 5 and num_tokens < 200 and id_count > 5)
                    ) and at_masking
            ):
                tokens_new.append('@')
            else:
                if type(tok) == str:
                    tokens_new.append(tok)
                else:
                    tokens_new.append(tok.text)
            id_count += 1
        new_lines_filtered.append(' '.join(tokens_new))
        if found_long_token:
            break
    return '\n'.join(new_lines_filtered)

def apply_at_masking(text):
    tokens = tokenizer(text)
    tokens_new = []
    for i, tok in enumerate(tokens):
        if (
                (i % 100 < 7 and len(tokens) >= 200 and i > 7) or
                (i % 20 < 5 and len(tokens) < 200 and i > 5)
        ):
            tokens_new.append('@')
        else:
            tokens_new.append(tok.text)
    return ' '.join(tokens_new)

def remove_tokens_func(text, tokens):
    for token in tokens:
        text = text.replace(token, '')
    return text

def main(
    path_to_prompt_data="data/nela-covid-2020", # pristine data alternative is webtext
    path_to_generated_data="data/generated", # generated data folder
    path_to_gentext_folder='gentext_data', # destination folder
    # model_names=['gptneo-covid', 'gpt2-3-covid', 'gpt2-md-covid', 'gpt2-lg-covid', 'gpt2-xl-covid'],
    # decodings=['topp96', 'topk40', 'random'],
    # model_names=['gptneo-covid'],
    # model_names=['gpt-neo-2.7B-xl-1542M-nucleus', 'gpt-j-6b-xl-1542M-nucleus'],
    model_names=['gpt3-davinci-002-webtext'],
    decodings=['topp96'],
    # decodings=['topp96', 'topk40', 'random'],
    # splits=['valid', 'test','train'],
    splits=['test'],
    keep_as_is=False,
    enforce_balanced=True,
    remove_tokens=['<|endoftext|>'],
):
    if type(model_names) == str:
        model_names = [model_names]
    for model_name in model_names:
        for decoding in decodings:
            for split in splits:
                print(f'Starting {model_name}, {decoding}, {split}')
                # Load pristine
                if os.path.exists(f'{path_to_prompt_data}-{split}.json'):
                    with open(f'{path_to_prompt_data}-{split}.json') as infile:
                        pristine = [json.loads(line.strip()) for line in infile.readlines()]
                else:
                    with open(f'{path_to_prompt_data}.{split}.jsonl') as infile:
                        pristine = [json.loads(line.strip()) for line in infile.readlines()]
                
                # Load generated
                if os.path.exists(f'{path_to_generated_data}/{model_name}-{split}-{decoding}.json'):
                    with open(f'{path_to_generated_data}/{model_name}-{split}-{decoding}.json') as infile:
                        generated = [json.loads(line.strip()) for line in infile.readlines()]
                else:
                    with open(f'{path_to_generated_data}/{model_name}-{decoding}.{split}.json') as infile:
                        generated = [json.loads(line.strip()) for line in infile.readlines()]
                
                if enforce_balanced:
                    if len(pristine) > len(generated):
                        print('WARNING: len pristine greater than generated, truncating to equalize')
                        pristine = pristine[:len(generated)]
                    elif len(generated) > len(pristine):
                        print('WARNING: len generated greater than pristine, truncating to equalize')
                        generated = generated[:len(pristine)]
                    assert len(generated) == len(pristine)

                # add labels to data
                for elt in pristine:
                    elt['label'] = 'pristine'
                    text = remove_tokens_func(elt['text'], remove_tokens)
                    if not keep_as_is:
                        elt['text_untokenized'] = elt['text']
                        elt['text'] = apply_at_masking_and_filter(text, at_masking=False)
                    else:
                        elt['text'] = text
                
                # filter the generated data and add the @ signs
                for elt in tqdm(generated):
                    elt['label'] = 'generated'
                    generation = remove_tokens_func(elt['generation'], remove_tokens)
                    if not keep_as_is:
                        elt['text_unfiltered'], elt['prompt_text'] = elt['generation'], elt['text']
                        filtered_at_text = apply_at_masking_and_filter(generation)
                        elt['text'] = filtered_at_text
                    else:
                        elt['text'], elt['prompt_text'] = generation, elt['text']
                    del elt['generation']

                # shuffle
                data = pristine + generated
                random.shuffle(data)

                # Write to the destination
                with open(f'{path_to_gentext_folder}/{model_name}-{decoding}-v2.{split}.jsonl', 'w') as outfile:
                    c = 0
                    for elt in data:
                        c += 1
                        outfile.write(f'{json.dumps(elt)}\n')
                    print(f'Wrote {c} lines to file.')
                
                print(f'Ending {model_name}, {decoding}, {split}')

if __name__ == '__main__':
    fire.Fire(main)
