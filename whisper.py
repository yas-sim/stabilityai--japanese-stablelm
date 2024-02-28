import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from datasets import load_dataset

import os
import openvino as ov
import openvino.properties.hint as hints

device='GPU.1'

model_id = "openai/whisper-large-v3"
ov_config = {
    'CACHE_DIR':'./cache', 
    'PERFORMANCE_HINT':'LATENCY',
    'NUM_STREAMS':'1', 
    'GPU_QUEUE_THROTTLE': 'LOW',
}
if not os.path.exists('whisper_model'):
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id, export=True, compile=False
    )
    model.half()
    model.save_pretrained('whisper_model')
else:
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        './whisper_model/', cache_dir='./cache', device=device, ov_config=ov_config
    )

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=1,
    #return_timestamps=True,
    #torch_dtype=torch_dtype,
    #device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
