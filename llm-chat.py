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
model="google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    # to run the model on CPU wen need to ignore the quantization in the following line 
#    model_kwargs={"load_in_8bit": True},
    device=0 if torch.cuda.is_available() else -1
)


@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    if 'messages' not in data:
        return jsonify({'error': 'mesages array not provided'}), 400

    messages = data['messages']
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.9, top_k=10, top_p=0.95)
    generated_output = result[0]['generated_text']
    print(generated_output)
    return jsonify({'generated_text': generated_output})

if __name__ == '__main__':
    serve(app,  port=5050)
