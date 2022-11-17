import json
import os
from tqdm import tqdm
import random

random.seed(0)

base_path = "data/nela-covid-2020/newsdata/"

all_text = []
for fname in tqdm(os.listdir(base_path)):
    with open(base_path + fname) as infile:
        source_data = json.load(infile)
    for elt in source_data:
        elt['text'] = elt['content']
        del elt['content']
    
    all_text += source_data

random.shuffle(all_text)
train_text = all_text[:int(len(all_text)*0.95)]
valid_text = all_text[int(len(all_text)*0.95):]


with open("data/nela-covid-2020-train.json", 'w') as outfile:
    for elt in train_text:
        outfile.write(f'{json.dumps(elt)}\n')
with open("data/nela-covid-2020-valid.json", 'w') as outfile:
    for elt in valid_text:
        outfile.write(f'{json.dumps(elt)}\n')

print(f"wrote {len(all_text)} articles to data/nela-covid-2020-train/valid.json")

with open('/data/artidoro/data/nela-covid-2020-valid.json') as infile:
    lines = [line.strip() for line in infile.readlines()]

valid = lines[:3000]
test = lines[3000:]

with open('/data/artidoro/data/nela-covid-2020-valid.json', 'w') as outfile:
    for line in valid:
        outfile.write(f'{line}\n')

with open('/data/artidoro/data/nela-covid-2020-test.json', 'w') as outfile:
    for line in test:
        outfile.write(f'{line}\n')