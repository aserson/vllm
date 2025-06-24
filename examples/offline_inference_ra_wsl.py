import time
from vllm import LLM, SamplingParams

# System prompts and system schema
sys_prompt = (
    "Sarah is a talented software developer working at a tech company in the city. "
    "Every morning, she follows a comforting routine that helps her prepare mentally for the day ahead. "
    "She wakes up early, around 6:30 AM, and spends some time stretching and doing light exercises. "
    "After that, she takes a quick shower and dresses in comfortable but professional clothes suitable for work. "
    "She leaves her apartment at 7:15 AM and walks to the same cozy coffee shop near her building. "
    "The coffee shop is a small place with a warm atmosphere, familiar faces, and the aroma of freshly ground coffee beans. "
    "Sarah always orders a cappuccino and a freshly baked croissant, which the barista knows by heart. "
    "The barista has greeted her by name for the past three years and often chats with her about the weather, new movies, or weekend plans. "
    "Sometimes, other regular customers join their light conversations, creating a friendly community vibe. "
    "After finishing her breakfast, Sarah enjoys a fifteen-minute walk to her office, listening to her favorite podcasts or soothing music. "
    "This quiet time helps her focus and sets a positive tone for the workday. "
    "At work, Sarah feels motivated and productive, tackling complex programming challenges with creativity and precision. "
    "She believes that these small daily habits, from exercise to coffee to music, make a significant difference in her overall well-being and career success."
)

# User prompts.
questions = [
    "What time does Sarah usually wake up in the morning?",
    "What does Sarah do before leaving her apartment?",
    "How long has the barista known Sarah and what do they usually talk about?",
    "What does Sarah listen to during her walk to the office?",
    "How does Sarah feel about her daily habits and their impact on her work?",
    "What does Sarah usually wear to work?",
    "How does Sarah feel about the atmosphere in the coffee shop?",
    "Who sometimes joins Sarah and the barista in their conversations?",
    "What kind of challenges does Sarah tackle at work?",
    "Why does Sarah believe small daily habits are important?"
]

# System schema
sys_schema = "[INST] <<SYS>>\n{__SYS_PROMPT}\n<</SYS>>\n\n{__USR_PROMPT} [/INST]"

relay_attention = True

# Create an LLM with system prompt
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
          enforce_eager=True,
          enable_relay_attention=relay_attention,
          max_model_len=512,
          gpu_memory_utilization=0.95)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

start_time = time.time()

if relay_attention:
    prompts = [q for q in questions]
    # prompts += ["Hi " + q for q in questions]
    llm.fill_sys_prompt(sys_prompt)
else:
    prompts = [sys_schema.format(__SYS_PROMPT=sys_prompt, __USR_PROMPT=question) for question in questions]
    # prompts += [sys_schema.format(__SYS_PROMPT=sys_prompt, __USR_PROMPT="Hi " + question) for question in questions]

# Generate the outputs
outputs = llm.generate(prompts, sampling_params)

end_time = time.time()
elapsed = end_time - start_time

# Print the outputs.
for output in outputs:
    index1 = output.prompt.find("<</SYS>>\n\n")
    index2 = output.prompt.find("[/INST]")
    if index1 == -1:
        if index2 != -1:
            prompt = output.prompt[:index2]
        else:
            prompt = output.prompt
    else:
        if index2 != -1:
            prompt = output.prompt[10 + index1:index2]
        else:
            prompt = output.prompt[10 + index1:]

    generated_text = output.outputs[0].text[2:]
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")

print(f"Generation took {elapsed:.3f} seconds")
