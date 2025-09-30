from vllm import LLM, SamplingParams

model_path = "/home/aistudio/config_folder" 

llm = LLM(
    model=model_path, 
    tensor_parallel_size=8,
    trust_remote_code=True,
    enforce_eager=True 
)

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")