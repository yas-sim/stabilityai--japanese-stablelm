# Run an LLM chat model only with OpenVINO (supports only the stateful, KV-caching enabled LLM models)
#  - Without 'optimum-intel', 'PyTorch' and HF-Tokenizers.
#  This program uses sampling method to generate the output text.

import numpy as np
from transformers import LlamaTokenizer
import openvino as ov

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

device = 'CPU'
ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache"}
print('Compiling the model...', end='', flush=True)
compiled_model = ov.compile_model('openvino_model_int4asym.xml', device, ov_config)
infer_request = compiled_model.create_infer_request()
print('finished.')
print(compiled_model)

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

# Tokenize the input text (text -> token IDs)
# - The model input for the 1st iteration
input_ids      = tokens.input_ids
attention_mask = tokens.attention_mask
position       = input_ids.shape[-1]
position_ids   = np.array([range(position)], dtype=np.int64)
beam_idx       = np.array([0], dtype=np.int32)

infer_request.reset_state()                                     # Initialize model internal state

num_max_token_for_generation = 20
generated_text_ids = []
prev_output = ''

past_key_values = ov.Tensor(type=ov.Type.f32 , shape=(32,2,1,32,0,128))

past_kv_names = []
for n in range(32):
    #past_kv_name = 'attn_weights'
    #if n!=0:
    #    past_kv_name += f'.{(n-1)*2+1}'
    past_kv_name = str(7237 + n)
    past_kv_names.append(past_kv_name)

for i in range(num_max_token_for_generation):

    # Run inference (to generate the logits for the next word prediction)
    #infer_request.reset_state()                                     # Initialize model internal state
    response = infer_request.infer(
        inputs={'input_ids'      : input_ids,
                'attention_mask' : attention_mask,
                'position_ids'   : position_ids,
                'past_key_values': past_key_values,
                'use_cache'      : [1],
                'output_attentions' : [1],
                'return_dict'    : [0],
        }
    )
    print(response)
    print(response[past_kv_names[0]].shape)  # [?,32,?,?]
    #(1, 53, 4096)
    #(1, 53, 4096)
    #(1, 32, 53, 53)

    logits_name = '7270'
    logits = response[logits_name][0, -1, :].ravel()
    sampled_token_id = np.argmax(logits)
    #print(logits, logits.shape, sampled_id)

    if sampled_token_id == tokenizer.eos_token_id:
        print('\n*** EOS token detected.')
        break
    generated_text_ids = np.append(generated_text_ids, sampled_token_id)  # Append the predicted word to the bottom of the generated text ID array
    output_text = tokenizer.decode(generated_text_ids)              # Decode and generate the text from the array of token IDs
    print(output_text[len(prev_output):], end='', flush=True)       # Print only the last generated word
    prev_output = output_text
    #print(output_text)

    #input_ids = np.append(input_ids, [[sampled_token_id]], axis=1)
    #attention_mask = np.append(attention_mask, [[1]], axis=1)
    #position_ids = np.append(position_ids, [[position]], axis=1)

    past_key_values = []
    for past_kv_name in past_kv_names:
        past_key_value = response[past_kv_name]
        past_key_values.append([past_key_value, past_key_value])

    input_ids      = np.array([[sampled_token_id]], dtype=np.int64)
    attention_mask = np.array([[1]], dtype=np.int64)
    position_ids   = np.array([[position]], dtype=np.int64)
    past_key_values = np.array(past_key_values)
    position      += 1
    print(past_key_values.shape)
    print()

print(f'\n\n*** Completed.')

"""
<Model: 'Model2'
inputs[
<ConstOutput: names[input_ids] shape[?,?] type: i32>,
<ConstOutput: names[attention_mask] shape[?,?] type: i32>,
<ConstOutput: names[position_ids] shape[?,?] type: i32>,
<ConstOutput: names[past_key_values] shape[?,?,?,?,?,?] type: f32>,
<ConstOutput: names[use_cache] shape[?] type: i32>,
<ConstOutput: names[output_attentions] shape[?] type: i32>,
<ConstOutput: names[output_hidden_states] shape[?] type: i32>,
<ConstOutput: names[return_dict] shape[?] type: i32>
]
outputs[
<ConstOutput: names[7270, 76] shape[?,?,65536] type: f32>,
<ConstOutput: names[11, 189, hidden_states.1, 7205] shape[?,?,4096] type: f32>,
<ConstOutput: names[437, hidden_states.3, 439, 7206, 12] shape[?,?,4096] type: f32>,
<ConstOutput: names[655, 7207, 657, hidden_states.5, 13] shape[?,?,4096] type: f32>,
<ConstOutput: names[873, hidden_states.7, 875, 7208, 14] shape[?,?,4096] type: f32>,
<ConstOutput: names[1091, hidden_states.9, 1093, 7209, 15] shape[?,?,4096] type: f32>,
<ConstOutput: names[1309, hidden_states.11, 1311, 7210, 16] shape[?,?,4096] type: f32>,
<ConstOutput: names[1527, 7211, hidden_states.13, 1529, 17] shape[?,?,4096] type: f32>,
<ConstOutput: names[1745, 1747, hidden_states.15, 7212, 18] shape[?,?,4096] type: f32>,
<ConstOutput: names[1963, hidden_states.17, 1965, 7213, 19] shape[?,?,4096] type: f32>,
<ConstOutput: names[20, 2181, hidden_states.19, 2183, 7214] shape[?,?,4096] type: f32>,
<ConstOutput: names[2401, 2399, hidden_states.21, 7215, 21] shape[?,?,4096] type: f32>,
<ConstOutput: names[22, 2617, hidden_states.23, 7216, 2619] shape[?,?,4096] type: f32>,
<ConstOutput: names[2835, hidden_states.25, 2837, 7217, 23] shape[?,?,4096] type: f32>,
<ConstOutput: names[3053, hidden_states.27, 3055, 7218, 24] shape[?,?,4096] type: f32>,
<ConstOutput: names[3271, hidden_states.29, 7219, 3273, 25] shape[?,?,4096] type: f32>,
<ConstOutput: names[3489, hidden_states.31, 3491, 7220, 26] shape[?,?,4096] type: f32>,
<ConstOutput: names[27, 3707, 3709, hidden_states.33, 7221] shape[?,?,4096] type: f32>,
<ConstOutput: names[3925, hidden_states.35, 3927, 7222, 28] shape[?,?,4096] type: f32>,
<ConstOutput: names[4143, hidden_states.37, 4145, 7223, 29] shape[?,?,4096] type: f32>,
<ConstOutput: names[4361, 7224, hidden_states.39, 4363, 30] shape[?,?,4096] type: f32>,
<ConstOutput: names[4579, 7225, 4581, hidden_states.41, 31] shape[?,?,4096] type: f32>,
<ConstOutput: names[32, 4797, hidden_states.43, 4799, 7226] shape[?,?,4096] type: f32>,
<ConstOutput: names[5015, hidden_states.45, 5017, 7227, 33] shape[?,?,4096] type: f32>,
<ConstOutput: names[5233, hidden_states.47, 7228, 5235, 34] shape[?,?,4096] type: f32>,
<ConstOutput: names[5451, 35, hidden_states.49, 5453, 7229] shape[?,?,4096] type: f32>,
<ConstOutput: names[7230, 5669, hidden_states.51, 5671, 36] shape[?,?,4096] type: f32>,
<ConstOutput: names[5887, hidden_states.53, 5889, 7231, 37] shape[?,?,4096] type: f32>,
<ConstOutput: names[6105, hidden_states.55, 7232, 6107, 38] shape[?,?,4096] type: f32>,
<ConstOutput: names[6323, 7233, hidden_states.57, 6325, 39] shape[?,?,4096] type: f32>,
<ConstOutput: names[7234, 6541, hidden_states.59, 40, 6543] shape[?,?,4096] type: f32>,
<ConstOutput: names[6759, hidden_states.61, 6761, 7235, 41] shape[?,?,4096] type: f32>,
<ConstOutput: names[6977, 6979, hidden_states, 7236, 42] shape[?,?,4096] type: f32>,
<ConstOutput: names[7202, 43, 7204] shape[?,?,4096] type: f32>,

<ConstOutput: names[44, 406, 407, 420, 440, 7237] shape[?,32,?,?] type: f32>,
<ConstOutput: names[7238, 624, 625, 658, 638, 45] shape[?,32,?,?] type: f32>,
<ConstOutput: names[842, 843, 856, 876, 7239, 46] shape[?,32,?,?] type: f32>,
<ConstOutput: names[1060, 1061, 1074, 1094, 7240, 47] shape[?,32,?,?] type: f32>,
<ConstOutput: names[1278, 1279, 1312, 1292, 48, 7241] shape[?,32,?,?] type: f32>,
<ConstOutput: names[1496, 1497, 1510, 1530, 7242, 49] shape[?,32,?,?] type: f32>,
<ConstOutput: names[1714, 1715, 1728, 1748, 7243, 50] shape[?,32,?,?] type: f32>,
<ConstOutput: names[1932, 1933, 1946, 1966, 7244, 51] shape[?,32,?,?] type: f32>,
<ConstOutput: names[2150, 2151, 2164, 52, 2184, 7245] shape[?,32,?,?] type: f32>,
<ConstOutput: names[2382, 2368, 2369, 2402, 53, 7246] shape[?,32,?,?] type: f32>,
<ConstOutput: names[2586, 2587, 7247, 2600, 2620, 54] shape[?,32,?,?] type: f32>,
<ConstOutput: names[2804, 2805, 7248, 2818, 2838, 55] shape[?,32,?,?] type: f32>,
<ConstOutput: names[56, 3022, 3023, 3056, 3036, 7249] shape[?,32,?,?] type: f32>,
<ConstOutput: names[3240, 3241, 3254, 7250, 3274, 57] shape[?,32,?,?] type: f32>,
<ConstOutput: names[3472, 3458, 3459, 3492, 7251, 58] shape[?,32,?,?] type: f32>,
<ConstOutput: names[3690, 3676, 3677, 3710, 7252, 59] shape[?,32,?,?] type: f32>,
<ConstOutput: names[3928, 60, 3894, 3895, 7253, 3908] shape[?,32,?,?] type: f32>,
<ConstOutput: names[7254, 4112, 4113, 4126, 4146, 61] shape[?,32,?,?] type: f32>,
<ConstOutput: names[4330, 4331, 4344, 4364, 7255, 62] shape[?,32,?,?] type: f32>,
<ConstOutput: names[63, 4548, 4549, 4562, 4582, 7256] shape[?,32,?,?] type: f32>,
<ConstOutput: names[4766, 4767, 4780, 7257, 4800, 64] shape[?,32,?,?] type: f32>,
<ConstOutput: names[4984, 4985, 4998, 5018, 7258, 65] shape[?,32,?,?] type: f32>,
<ConstOutput: names[5202, 5203, 5236, 5216, 7259, 66] shape[?,32,?,?] type: f32>,
<ConstOutput: names[5420, 5421, 5434, 5454, 7260, 67] shape[?,32,?,?] type: f32>,
<ConstOutput: names[5638, 5639, 5652, 7261, 5672, 68] shape[?,32,?,?] type: f32>,
<ConstOutput: names[5856, 5857, 69, 5870, 5890, 7262] shape[?,32,?,?] type: f32>,
<ConstOutput: names[6108, 6074, 6075, 6088, 7263, 70] shape[?,32,?,?] type: f32>,
<ConstOutput: names[6326, 7264, 6292, 6293, 6306, 71] shape[?,32,?,?] type: f32>,
<ConstOutput: names[6510, 6511, 6524, 6544, 72, 7265] shape[?,32,?,?] type: f32>,
<ConstOutput: names[6742, 7266, 6728, 6729, 6762, 73] shape[?,32,?,?] type: f32>,
<ConstOutput: names[6960, 74, 6946, 6947, 6980, 7267] shape[?,32,?,?] type: f32>,
<ConstOutput: names[7164, 7165, 7198, 7178, 7268, 75] shape[?,32,?,?] type: f32>
]>
"""