import time
from utils import set_env


set_env()
# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# For reproducibility
random.seed(0)
torch.manual_seed(0)

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
config.attention_dropout=None
model_name='meta-llama/Llama-2-7b-hf'
model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')
original_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()
for _ in range(5):
    st=time.time()
    output2= original_model.generate(inputs, max_new_tokens=96)
    ed1=time.time()
    output1 = model.generate(inputs, max_new_tokens=96)
    ed2=time.time()
    print('new/old time', ed2-ed1, ed1-st)
# config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

# # print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
# output1=enc.decode(output1[0].tolist()[inputs.shape[1]:], skip_special_tokens=True)
# output2=enc.decode(output2[0].tolist()[inputs.shape[1]:], skip_special_tokens=True)
# print(output1, "="*50,
#       output2)