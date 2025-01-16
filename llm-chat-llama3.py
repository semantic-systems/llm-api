from flask import Flask, request, jsonify
import transformers
import torch
import os
from waitress import serve
import logging
from flask_swagger_ui import get_swaggerui_blueprint
import json

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0 if torch.cuda.is_available() else -1
)

# Define default values for parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_MAX_GEN_LEN = 512

@app.route('/generate_text', methods=['POST'])
def generate_text():
    """
    Generate Text
    ---
    post:
      summary: Generate text based on input messages
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                messages:
                  type: array
                  items:
                    type: object
                    properties:
                      role:
                        type: string
                        default: "system"
                        description: "The role of the message (e.g., system, user)"
                      content:
                        type: string
                        default: "Utilize prompt engineering to classify the given text accurately into one of the following predefined categories:\\n    Environment\\n    Soziales\\n    Governance\\n    Keine Armut\\n    Kein Hunger\\n    E-Umweltschutz\\n    E-Klimaschutz\\n    E-Erneuerbare Energie\\n    E-Emissionsreduktion\\n    E-Ressourceneffizienz\\n    S-Arbeitssicherheit\\n    S-Gesundheitsschutz\\n    S-Arbeitsbedingungen\\nLimit your response to the identified class, nothing else. Optimize for increased accuracy."
                        description: "The content of the message"
                  description: "List of input messages"
                  default:
                    - role: "system"
                      content: "Utilize prompt engineering to classify the given text accurately into one of the following predefined categories:\\n    Environment\\n    Soziales\\n    Governance\\n    Keine Armut\\n    Kein Hunger\\n    E-Umweltschutz\\n    E-Klimaschutz\\n    E-Erneuerbare Energie\\n    E-Emissionsreduktion\\n    E-Ressourceneffizienz\\n    S-Arbeitssicherheit\\n    S-Gesundheitsschutz\\n    S-Arbeitsbedingungen\\nLimit your response to the identified class, nothing else. Optimize for increased accuracy."
                    - role: "user"
                      content: "Am Wochenende fand ein tolles soziales Event statt. Die Gemeinschaft sammelte Spenden für wohltätige Zwecke und stärkte das Bewusstsein für lokale Anliegen. Mit Musik, Essen und Aktivitäten für alle Altersgruppen war die Veranstaltung ein großer Erfolg und förderte den Zusammenhalt der Gemeinde."
                temperature:
                  type: number
                  default: 0.7
                  description: Sampling temperature (optional)
                top_p:
                  type: number
                  default: 0.9
                  description: Nucleus sampling parameter (optional)
                max_new_tokens:
                  type: integer
                  default: 256
                  description: Maximum number of new tokens to generate (optional)
                max_seq_len:
                  type: integer
                  default: 1024
                  description: Maximum sequence length (optional)
                max_gen_len:
                  type: integer
                  default: 512
                  description: Maximum generation length (optional)
      responses:
        200:
          description: Generated text
          content:
            application/json:
              schema:
                type: object
                properties:
                  generated_text:
                    type: string
        400:
          description: Bad Request
        500:
          description: Internal Server Error
    """
    data = request.json

    # Log the incoming request
    logger.info(f"Incoming request data: {json.dumps(data)}")

    if 'messages' not in data:
        error_message = 'The "messages" key is required. Optional parameters: temperature, top_p, max_new_tokens, max_seq_len, max_gen_len.'
        logger.error(error_message)
        return jsonify({'error': error_message}), 400

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

        # eos_token_id or terminators
        eos_token_id = pipeline.tokenizer.eos_token_id
    
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

        # Log the generated response
        logger.info(f"Generated text: {result}")

        return jsonify({'generated_text': result})
    
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Text Generation and Classification API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Ensure the static directory exists
os.makedirs('static', exist_ok=True)

# Generate Swagger JSON
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Text Generation and Classification API",
        "description": "API for generating text based on input messages using a pre-trained large language model.",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["https"],
    "paths": {
        "/generate_text": {
            "post": {
                "summary": "Generate text based on input messages",
                "consumes": ["application/json"],
                "produces": ["application/json"],
                "parameters": [
                    {
                        "in": "body",
                        "name": "body",
                        "required": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "messages": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "role": {
                                                "type": "string",
                                                "default": "system"
                                            },
                                            "content": {
                                                "type": "string",
                                                "default": (
                                                    "Utilize prompt engineering to classify the given text accurately into one of the following predefined categories:\n"
                                                    "    Environment\n"
                                                    "    Soziales\n"
                                                    "    Governance\n"
                                                    "    Keine Armut\n"
                                                    "    Kein Hunger\n"
                                                    "    E-Umweltschutz\n"
                                                    "    E-Klimaschutz\n"
                                                    "    E-Erneuerbare Energie\n"
                                                    "    E-Emissionsreduktion\n"
                                                    "    E-Ressourceneffizienz\n"
                                                    "    S-Arbeitssicherheit\n"
                                                    "    S-Gesundheitsschutz\n"
                                                    "    S-Arbeitsbedingungen\n"
                                                    "Limit your response to the identified class, nothing else. Optimize for increased accuracy."
                                                )
                                            }
                                        },
                                        "required": ["role", "content"]
                                    },
                                    "description": "List of input messages",
                                    "default": [
                                        {
                                            "role": "system",
                                            "content": (
                                                "Utilize prompt engineering to classify the given text accurately into one of the following predefined categories:\n"
                                                "    Environment\n"
                                                "    Soziales\n"
                                                "    Governance\n"
                                                "    Keine Armut\n"
                                                "    Kein Hunger\n"
                                                "    E-Umweltschutz\n"
                                                "    E-Klimaschutz\n"
                                                "    E-Erneuerbare Energie\n"
                                                "    E-Emissionsreduktion\n"
                                                "    E-Ressourceneffizienz\n"
                                                "    S-Arbeitssicherheit\n"
                                                "    S-Gesundheitsschutz\n"
                                                "    S-Arbeitsbedingungen\n"
                                                "Limit your response to the identified class, nothing else. Optimize for increased accuracy."
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": "Am Wochenende fand ein tolles soziales Event statt. Die Gemeinschaft sammelte Spenden für wohltätige Zwecke und stärkte das Bewusstsein für lokale Anliegen. Mit Musik, Essen und Aktivitäten für alle Altersgruppen war die Veranstaltung ein großer Erfolg und förderte den Zusammenhalt der Gemeinde."
                                        }
                                    ]
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7,
                                    "description": "Sampling temperature (optional)"
                                },
                                "top_p": {
                                    "type": "number",
                                    "default": 0.9,
                                    "description": "Nucleus sampling parameter (optional)"
                                },
                                "max_new_tokens": {
                                    "type": "integer",
                                    "default": 256,
                                    "description": "Maximum number of new tokens to generate (optional)"
                                },
                                "max_seq_len": {
                                    "type": "integer",
                                    "default": 1024,
                                    "description": "Maximum sequence length (optional)"
                                },
                                "max_gen_len": {
                                    "type": "integer",
                                    "default": 512,
                                    "description": "Maximum generation length (optional)"
                                }
                            },
                            "required": ["messages"]
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Generated text",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "generated_text": {"type": "string"}
                            }
                        }
                    },
                    "400": {"description": "Bad Request"},
                    "500": {"description": "Internal Server Error"}
                }
            }
        }
    }
}



with open('static/swagger.json', 'w') as f:
    json.dump(swagger_template, f)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5050)