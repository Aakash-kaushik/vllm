from vllm import SamplingParams, LLaVA

image = 'http://d1gpne0zfylr7s.cloudfront.net/b2e70b96-8ab7-4406-b3a8-a2597d220269_1706584896932.png'
llm = LLaVA(model="llava-hf/llava-1.5-7b-hf",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            enforce_eager=True)
prompts = 'USER: <image>\nWhat are these?\nASSISTANT:'

sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=256)
outputs = llm.generate(prompts, sampling_params,
                       images=image)  # PIL url or base64。
print(outputs)
