from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread

MODEL_PATH = r"Models\Dominator_Dolphin"
SYSTEM_PROMPT = """"""

def REDDIT_PROMPT(subreddit=None, context=None):
    if subreddit and context:
        instruction = f"You are a toxic user in r/{subreddit}.Reply to given comment in context: {context}"
        return instruction
    if subreddit:
        instruction = f"You are a toxic user in r/{subreddit}.Reply to given comment."
        return  instruction
    if context:
        instruction = f"You are a toxic user in r/AskReddit.Reply to given comment in context: {context}"
        return  instruction
    return "You are a toxic user in r/AskReddit.Reply to given comment."

def MATH_PROMPT(LaTeX=False, MCQ=False, Code=False, Explain=True, Comments=True, Asymptote=False):
    instruction = "You are a Math tutor, solve the following question."
    if MCQ:
        instruction = "You are a Math tutor, solve the following multiple choice question giving the correct option and explanation without Latex formatting"
    elif Asymptote:
        instruction = "You are a Math tutor, solve the following question giving a detailed explanation and asymptote diagram and Latex formatting"
    elif (not Code) and LaTeX and Explain:
        instruction = "You are a Math tutor, solve the following question giving a detailed explanation with Latex formatting"
    elif (not Code) and (not LaTeX) and Explain:
        instruction = "You are a Math tutor, solve the following question giving an explanation without Latex formatting"
    elif Code and (not LaTeX) and (not Explain) and (not Comments):
        instruction = "You are a math tutor, solve the following question in python without comments giving the final answer"
        # Gives "<TOOL_CALL>\n{OUTPUT}\n</TOOL_CALL>" as output
    elif Code and (not LaTeX) and (not Explain) and Comments:
        instruction = "You are a math tutor, solve the following question in python with comments giving the final answer"
        # Gives "<TOOL_CALL>\n{OUTPUT}\n</TOOL_CALL>" as output
    elif (not Code) and LaTeX and (not Explain):
        instruction = "You are a Math tutor, solve the following question giving a detailed explanation without Latex formatting"
    return instruction

print("⏳ Loading Dominator Dolphin...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0"
)
FastLanguageModel.for_inference(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>")
]

print("\n🗑️ DOMINATOR DOLPHIN ONLINE. (Type 'exit' to quit)")
print("--------------------------------------------------")

while True:
    try:
        user_input = input("\nYou: ")
    except EOFError:
        break

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Bye.")
        break

    #Set Temperature
    is_technical = any(
        word in user_input.lower() for word in ['code', 'python', 'script', 'solve', 'calculate', 'math', 'function'])
    current_temp = 0.1 if is_technical else 1.8

    maths = MATH_PROMPT(Code=True, LaTeX=False, Explain=False, Comments=False)
    current_temp = 0.1
    reddit = REDDIT_PROMPT(subreddit='RoastMe')
    messages = [
        {"role": "system", "content": maths},
        {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=current_temp,
        min_p=0.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Bot:", end=" ")

    for new_text in streamer:
        if "<|im_end|>" not in new_text:
            print(new_text, end="", flush=True)

    print()