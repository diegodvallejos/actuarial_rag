import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, BitsAndBytesConfig
from dotenv import load_dotenv, find_dotenv
import requests
import os

load_dotenv(find_dotenv())

# Load the model and processor from Hugging Face

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
headers = {"Authorization": f"Bearer {os.environ.get('MISTRAL_API_KEY')}"}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you tell me about yourself? ",
})


print(output)