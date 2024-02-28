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
prompt = 'Hello '
tokens = tokenizer(
    prompt, 
    add_special_tokens=False, 
    return_tensors="pt"
)

input_ids = tokens.input_ids
attention_mask = tokens.attention_mask
position_ids = torch.Tensor([ n for n in range(input_ids.shape[-1])])
print(input_ids.shape) # [1,2]


#print(input_ids, input_ids.shape)
#import sys
#sys.exit()

#print(tokens)
#print(dir(tokens))

res = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)

#print(res)

print(len(res['past_key_values']))  # 32
print(len(res['past_key_values'][0])) # 2
#print(len(res['past_key_values'][0][0].shape)) # 4
#print(len(res['past_key_values'][0][1].shape)) # 4
tensor = res['past_key_values'][0][0]
print(tensor.shape) # torch.Size([1, 32, 2, 128])
tensor = res['past_key_values'][0][1]
print(tensor.shape) # torch.Size([1, 32, 2, 128])

# logits
# past_key_values
# grad_fn = None
# loss = None
# hidden_states = None
# attention = None 
"""
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
"""


#tokens = model.generate(
#    input_ids,
##    max_new_tokens=256,
#    temperature=2.0,
#    top_p=0.95,
#    do_sample=True,
#)

#out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
#print(out)