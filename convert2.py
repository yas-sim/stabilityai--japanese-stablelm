import numpy as np
from transformers import AutoModelForCausalLM
import openvino as ov

import torch

model_vendor, model_name = 'stabilityai', 'japanese-stablelm-base-alpha-7b'

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

num_seq = 100
n = num_seq
# from config.json
vocab_size = 65536
hidden_size = 4096
num_hidden_layers = 32
num_attention_heads = 32

example_input = {
    'input_ids': torch.tensor([[123]], dtype=torch.int),
    'attention_mask': torch.tensor([[1]],dtype=torch.int),
    'position_ids': torch.tensor([[0]], dtype=torch.int),
    #'inputs_embeds': np.array([0], dtype=np.int32),
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    #'head_mask': torch.Tensor(size=(num_attention_heads,1,1,1,n)).to(torch.float32),

    'past_key_values': torch.Tensor(size=(1, 2, 1, num_attention_heads, num_seq, hidden_size // num_hidden_layers)).to(torch.float32), # hidden_size // num_hidden_layers
    #  past_key_values [n][0|1][ 1, 32, seq_len, 128]

    #'labels': np.zeros((1,100), dtype=np.int32),
    'use_cache': torch.tensor([1], dtype=torch.int32),
    'output_attentions': torch.tensor([32], dtype=torch.int32),
    'output_hidden_states': torch.tensor([32], dtype=torch.int32),
    'return_dict': torch.tensor([0], dtype=torch.int32),
}

print(example_input)

model = AutoModelForCausalLM.from_pretrained(f'{model_vendor}/{model_name}', trust_remote_code=True)
model.eval()

ov_model = ov.convert_model(model, example_input=example_input)

print(ov_model)
ov.save_model(ov_model, 'openvino_model.xml')

#from optimum.intel.openvino import OVModelForCausalLM
#model = OVModelForCausalLM.from_pretrained(f'{model_vendor}/{model_name}', trust_remote_code=True, export=True, compile=False, load_in_8bit=False)

"""
<Model: 'Model0'
inputs[
<ConstOutput: names[input_ids] shape[?,?] type: i32>,
<ConstOutput: names[attention_mask] shape[?,?] type: f32>,
<ConstOutput: names[position_ids] shape[?,?] type: i32>,
<ConstOutput: names[head_mask, 146, head_mask.3] shape[?,?,?,?,?] type: f32>,
<ConstOutput: names[use_cache] shape[?] type: i32>,
<ConstOutput: names[output_attentions] shape[?] type: i32>,
<ConstOutput: names[output_hidden_states] shape[?] type: i32>,
<ConstOutput: names[return_dict] shape[?] type: i32>
]
outputs[
<ConstOutput: names[6522] shape[?,?,65536] type: f32>
]>
"""