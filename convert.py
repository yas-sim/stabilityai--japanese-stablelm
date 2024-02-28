import numpy as np
from transformers import AutoModelForCausalLM
import inference as ov

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

example_input = {
    'input_ids': np.zeros((1,100,), dtype=np.int32),
    'attention_mask': np.zeros((1,100,), dtype=np.float32),
    'position_ids': np.zeros((1,100,), dtype=np.int32),
    #'inputs_embeds': np.array([0], dtype=np.int32),
    #'head_mask': np.zeros((32,1,1,1,100), dtype=np.float32),
    #'past_key_values': np.zeros((1,1,100), dtype=np.float32),
    #'labels': np.zeros((1,100), dtype=np.int32),
    'use_cache': np.array([1], dtype=np.int32),
    #'output_attentions': np.array([0], dtype=np.int32),
    #'output_hidden_states': np.array([0], dtype=np.int32),
    'return_dict': np.array([0], dtype=np.int32),
}

model = AutoModelForCausalLM.from_pretrained(f'{model_vendor}/{model_name}', trust_remote_code=True)
model.eval

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