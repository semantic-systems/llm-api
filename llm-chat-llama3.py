from flask import Flask, request, jsonify
import transformers
from transformers import AutoTokenizer
import torch
import os
import re
import sys
import json
import time
import random
import re
from waitress import serve


# the following line is to force the script to run on CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
#model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    # to run the model on CPU wen need to ignore the quantization in the following line 
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0 if torch.cuda.is_available() else -1
)


@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    if 'messages' not in data:
        return jsonify({'error': 'mesages array not provided'}), 400

    messages = data['messages']
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id,pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt,max_new_tokens=256,eos_token_id=terminators,do_sample=True,temperature=0.1,top_p=0.9)
    result = outputs[0]["generated_text"][len(prompt):]
    print("result:", result)
    generated_output = result
    return jsonify({'generated_text': generated_output})

if __name__ == '__main__':
    serve(app,  port=5050)
