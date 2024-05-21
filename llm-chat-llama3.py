from flask import Flask, request, jsonify
import transformers
import torch
import os
import logging
from waitress import serve

# Configure logging
logging.basicConfig(level=logging.INFO)

# Force the script to run on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Initialize the text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0 if torch.cuda.is_available() else -1
)

# Define default values for parameters
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_MAX_GEN_LEN = 512

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    
    # Validate the input
    if 'messages' not in data:
        return jsonify({'error': 'The "messages" key is required. Optional parameters: temperature, top_p, max_new_tokens, max_seq_len, max_gen_len.'}), 400

    # Retrieve parameters from the request or use default values
    temperature = data.get('temperature', DEFAULT_TEMPERATURE)
    top_p = data.get('top_p', DEFAULT_TOP_P)
    max_new_tokens = data.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
    max_seq_len = data.get('max_seq_len', DEFAULT_MAX_SEQ_LEN)
    max_gen_len = data.get('max_gen_len', DEFAULT_MAX_GEN_LEN)
    messages = data['messages']

    try:
        # Create the prompt using the tokenizer's chat template
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Handle eos_token_id and additional terminators if needed
        empty_token_id = pipeline.tokenizer.convert_tokens_to_ids("")
        if empty_token_id is not None and empty_token_id != pipeline.tokenizer.unk_token_id:
            eos_token_id = [pipeline.tokenizer.eos_token_id, empty_token_id]
        else:
            eos_token_id = [pipeline.tokenizer.eos_token_id]

        # Generate the text
        outputs = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract the generated text
        result = outputs[0]["generated_text"][len(prompt):]
        logging.info(f"Generated text: {result}")
        generated_output = result

        return jsonify({'generated_text': generated_output})
    
    except Exception as e:
        logging.error(f"Error during text generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    serve(app, port=5050)




# First version
"""# the following line is to force the script to run on CPU
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
    serve(app,  port=5050)"""
