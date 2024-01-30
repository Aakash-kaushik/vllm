from vllm import SamplingParams, LLaVA

image = 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Stonehenge.jpg'
llm = LLaVA(model="llava-hf/llava-1.5-7b-hf",  gpu_memory_utilization=0.95) 
prompts = [ 'prompt1',
'prompt2 <image> \n say',
'prompt3 <image> say something <image> something',
]
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)
outputs = llm.generate(prompts, sampling_params, images=[image]*3) # PIL url or base64。
