import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# https://h2o.ai/platform/mississippi/

# Set up the model and tokenizer
model_path = 'h2oai/h2ovl-mississippi-800m'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# config.llm_config._attn_implementation = 'flash_attention_2' # Removed explicit setting of attention implementation
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()

# Move the model to the GPU
model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=2048, do_sample=True)


# Example for single image
image_file = '/content/LP_USA_California_passenger.jpg'
question = '<image>\nRead the text in the image.'
response, history = model.chat(tokenizer, image_file, question, generation_config, history=None, return_history=False)
print(f'Result:  {response}')
