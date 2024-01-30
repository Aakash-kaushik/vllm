from vllm import SamplingParams, LLaVA

image = 'http://d1gpne0zfylr7s.cloudfront.net/d0d658cf-e799-4214-a26a-7aadd65087e0_1706578213306.png'
llm = LLaVA(model="llava-hf/llava-1.5-7b-hf", tensor_parallel_size=1,gpu_memory_utilization=0.95, enforce_eager=True) 
prompts = [ 'prompt1',
'prompt2 <image> \n say',
'prompt3 <image> say something <image> something',
]
sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=256)
outputs = llm.generate(prompts, sampling_params, images=[image]*3) # PIL url or base64。
