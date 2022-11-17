python generate_gpt3_api.py \
    --top_p=0.96 \
    --engine=text-davinci-002 \
    --path_to_prompt_data=gentext_data/webtext.train.jsonl \
    --path_to_generated_data=data/generated/gpt3-davinci-002-webtext-topp96.train.json \
    --max_num_docs 30000 \
    --batch_size 16 \
    # --debug_doc_nb 3