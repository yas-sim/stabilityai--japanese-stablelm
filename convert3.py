import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoConfig
from optimum.intel import OVModelForCausalLM

#tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

#model = AutoModelForCausalLM.from_pretrained(
#    "stabilityai/japanese-stablelm-instruct-alpha-7b",    
#    trust_remote_code=True,
#)

#model_id = 'stabilityai/japanese-stablelm-base-alpha-7b'
model_id = 'stabilityai/japanese-stablelm-base-gamma-7b'
model = OVModelForCausalLM.from_pretrained(
    model_id=model_id, 
    export=True, compile=False, load_in_8bit=False, trust_remote_code=True, 
    config=AutoConfig.from_pretrained(model_id, trust_remote_code=True)
)

model.half()
model.save_pretrained('ovmodel')

import sys
sys.exit()
#model.eval()

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

input_ids = tokenizer.encode(
    prompt, 
    add_special_tokens=False, 
    return_tensors="pt"
)

tokens = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=2.0,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
print(out)