import numpy as np
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

model = AutoModelForCausalLM.from_pretrained(
    #"stabilityai/japanese-stablelm-instruct-alpha-7b",    
    "stabilityai/japanese-stablelm-base-alpha-7b",    
    trust_remote_code=True,
)
#model.half()
model.eval()

def build_prompt(user_query, inputs="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + user_query, ": "]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

# this is for reproducibility.
# feel free to change to get different result
seed = 42
torch.manual_seed(seed)

# Infer with prompt without any additional input
user_inputs = {
    "user_query": "VR とはどのようなものですか？",
    "inputs": ""
}
prompt = build_prompt(**user_inputs)

tokens = tokenizer(
    prompt, 
    add_special_tokens=False, 
    return_tensors="pt"
)

pos = tokens.input_ids.shape[-1]

inputs = {
    'input_ids'       : tokens.input_ids,
    'attention_mask'  : tokens.attention_mask,
    'position_ids'    : torch.tensor([n for n in range(pos)], dtype=torch.int64),
    'past_key_values' : None,
    'use_cache'       : True,
    'return_dict'     : True
}

num_generate_tokens = 20

for nn in range(num_generate_tokens):

    res = model(**inputs)

    # outpus
    #  logits [1, 3, 65536]
    #  past_key_values [n][0|1][ 1, 32, seq_len, 128]

    past_key_values = res['past_key_values']
    logits = res['logits'][0,-1,:].to('cpu').detach().numpy().copy()
    predicted_id = np.argmax(logits)
    if predicted_id == tokenizer.eos_token_id:
        print('** EOS token is generated.')
        break
    token = tokenizer.decode([predicted_id], skip_special_tokens=False)
    print(predicted_id, token, res['past_key_values'][0][0].shape)

    inputs = {
        'past_key_values' : past_key_values,
        'input_ids'       : torch.tensor([[predicted_id]], dtype=torch.int64),
        'attention_mask'  : torch.tensor([[1]], dtype=torch.int64),
        'position_ids'    : torch.tensor([[pos]], dtype=torch.int64),
        'use_cache'       : True,
        'return_dict'     : True
    }
    pos += 1
