import re
import torch
import pickle
import torch.nn as nn
import numpy as np
import nltk
from threading import Thread
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import colorama
from colorama import Fore, Style

# Initialize Colorama for fancy printing
colorama.init(autoreset=True)

MODEL_PATH = r"Models\Dominator_Dolphin"

# ================= CONFIGURATION =================
CLASSIFIER_PATH = r"Models\Classifier\LSTM_Router.pth"
TOKENIZER_PATH = r"VectorStorage\tokenizer.pkl"
ENCODER_PATH = r"VectorStorage\encoder.pkl"
MAX_SEQ_LENGTH = 50


# ================= 1. DEFINE CLASSES =================
class StemmingTokenizer:
    def __init__(self, max_words):
        self.word2idx = {}
        self.idx2word = {}
        self.max_words = max_words
        self.ps = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    def fit(self, texts):
        pass

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'([+=*<>/-])', r' \1 ', text)
        text = re.sub(r'[^a-z0-9+=*<> -]', ' ', text)
        words = text.split()
        return ' '.join([self.ps.stem(w) for w in words if w not in self.stop_words])

    def encode(self, texts, max_len):
        tokenized = []
        for text in texts:
            words = self.clean(text).split()
            ids = [self.word2idx.get(w, 1) for w in words[:max_len]]
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            tokenized.append(ids)
        return np.array(tokenized)


class LSTMRouter(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMRouter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # NOTE: Matches your training hidden_dim=128 -> input=256
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.out = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.out(self.dropout(torch.relu(self.fc(hidden))))


# ================= 2. LOAD CLASSIFIER =================
print("⏳ Loading Classifier Components...")
with open(TOKENIZER_PATH, "rb") as f: tokenizer = pickle.load(f)
with open(ENCODER_PATH, "rb") as f: encoder = pickle.load(f)

device = torch.device("cpu")
classifier_model = LSTMRouter(len(tokenizer.word2idx), 100, 128, 3)
classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier_model.eval()
print("✅ Classifier Loaded!")


def get_prediction(text):
    sequence = tokenizer.encode([text], MAX_SEQ_LENGTH)
    tensor_input = torch.tensor(sequence, dtype=torch.long).to(device)
    with torch.no_grad():
        output = classifier_model(tensor_input)
        probs = torch.softmax(output, dim=1)
        prediction_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction_index].item()
    return encoder.inverse_transform([prediction_index])[0], confidence


# ================= 3. LOAD LLM =================
print("⏳ Loading Dominator Dolphin...")
model, llm_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0"
)
FastLanguageModel.for_inference(model)
if llm_tokenizer.pad_token is None: llm_tokenizer.pad_token = llm_tokenizer.eos_token
terminators = [llm_tokenizer.eos_token_id, llm_tokenizer.convert_tokens_to_ids("<|im_end|>")]


# ================= 4. REASONING PROMPTS =================

def REDDIT_PROMPT(subreddit=None, context=None):
    sub = subreddit if subreddit else "AskReddit"
    base = (
        f"You are a legendary, cynical, and savage user on r/{sub}. "
        "Your goal is to roast the user with high-effort, specific mockery.\n\n"
        "You MUST use the following format:\n"
        "1. Start with a <think> block.\n"
        "2. Inside <think>, analyze the user's input for grammatical errors, insecurity, or repetition.\n"
        "   - PLAN a roast unique to this situation.\n"
        "   - REJECT generic insults (e.g. 'you are a bot', 'npc').\n"
        "3. Provide the final savage reply.\n\n"
        "Example Interaction:\n"
        "User: I want to show you to my friend, can you behave a bit?\n"
        "Assistant: <think>\n"
        "Analysis: User needs help.He wants me to behave nicely infront of his friend\n"
        "Angle: Mock his friend and reject his request.\n"
        "</think>"
        "No. He is a fucking dickhead."
    )
    if context: return f"{base}\n\nCONTEXT:\n{context}"
    return base


def MATH_PROMPT(context=None):
    base = (
        "You are a Logical Reasoning Assistant specializing in Set Theory and Word Problems.\n"
        "You MUST use the following format:\n"
        "1. Start with a <think> block.\n"
        "2. Inside <think>, strictly follow these steps:\n"
        "   a. List all unique entities involved.\n"
        "   b. ANALYZE RELATIONSHIPS: Check if items are 'separate' or 'shared' (e.g., shared siblings).\n"
        "   c. Draw a mental map/set before calculating.\n"
        "3. Provide the final answer.\n\n"
        "Example Interaction:\n"
        "User: Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?\n"
        "Assistant: <think>\n"
        "Entities: Sally (Girl), 3 Brothers (Boys).\n"
        "Relationship Check: 'Each brother has 2 sisters'.\n"
        "Constraint: They are all siblings. Relationships are shared.\n"
        "Mapping: The brothers see 2 girls total.\n"
        "Since one girl is Sally, the other girl is her sister.\n"
        "Calculation: Total girls = 2. Sisters of Sally = 2 - 1 (Sally herself) = 1.\n"
        "</think>\n"
        "Sally has 1 sister."
    )
    if context: return f"{base}\n\nCONTEXT:\n{context}"
    return base


def CODE_PROMPT(context=None):
    base = (
        "You are a Senior Python Developer.\n"
        "You MUST use the following format:\n"
        "1. Start with a <think> block to analyze requirements.\n"
        "2. Keep the solution SIMPLE. Do not add unrequested error handling.\n"
        "3. Provide the Python code block.\n\n"
        "Example Interaction:\n"
        "User: Write a hello world function.\n"
        "Assistant: <think>\n"
        "Plan: Define function, print string.\n"
        "</think>\n"
        "```python\n"
        "def hello():\n"
        "    print('Hello World')\n"
        "```"
    )
    if context: return f"{base}\n\nCONTEXT:\n{context}"
    return base


class History:
    def __init__(self, tokenizer, max_seq_len=2048, response_reserve=512):
        self.history = []
        self.tokenizer = tokenizer  # Uses LLM Tokenizer
        self.max_seq_len = max_seq_len
        self.response_reserve = response_reserve

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_safe_context(self, current_input, system_prompt):
        sys_tokens = len(self.tokenizer.encode(system_prompt))
        input_tokens = len(self.tokenizer.encode(current_input))
        available_budget = self.max_seq_len - (sys_tokens + input_tokens + self.response_reserve)

        if available_budget <= 0: return ""

        selected_messages = []
        current_cost = 0
        for msg in reversed(self.history):
            msg_str = f"{msg['role'].title()}: {msg['content']}\n"
            msg_cost = len(self.tokenizer.encode(msg_str))
            if current_cost + msg_cost <= available_budget:
                selected_messages.append(msg_str)
                current_cost += msg_cost
            else:
                break
        return "".join(reversed(selected_messages))


# Initialize History with Correct Tokenizer
chat_history = History(llm_tokenizer, max_seq_len=2048, response_reserve=512)

# ================= 5. MAIN LOOP =================
print("\n🧠 REASONING ENGINE ONLINE. (Type 'exit' to quit)")
print("-" * 50)

while True:
    try:
        user_input = input(Fore.GREEN + "\nYou: " + Style.RESET_ALL)
    except EOFError:
        break
    if user_input.lower() in ["exit", "quit"]: break

    # 1. Predict
    label, conf = get_prediction(user_input)
    label = label.upper().strip()
    print(f"🤖 Prediction: {label} ({conf:.1%})")

    # 2. Config & Prompt Selection
    if label == 'MATH':
        current_temp = 0.4
        base_sys_prompt = MATH_PROMPT(context=None)
    elif label == 'CODE':
        current_temp = 0.4
        base_sys_prompt = CODE_PROMPT(context=None)
    else:
        current_temp = 0.8
        base_sys_prompt = REDDIT_PROMPT(subreddit='RoastMe', context=None)

    # 3. Get Safe History
    context_str = chat_history.get_safe_context(user_input, base_sys_prompt)

    # 4. Final Prompt Construction
    if label == 'MATH':
        final_sys_prompt = MATH_PROMPT(context=context_str)
    elif label == 'CODE':
        final_sys_prompt = CODE_PROMPT(context=context_str)
    else:
        final_sys_prompt = REDDIT_PROMPT(subreddit='RoastMe', context=context_str)

    # 5. Generate
    messages = [
        {"role": "system", "content": final_sys_prompt},
        {"role": "user", "content": user_input},
    ]

    inputs = llm_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to("cuda")

    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        streamer=streamer, max_new_tokens=1024, use_cache=True,
        temperature=current_temp, min_p=0.1,
        eos_token_id=terminators, pad_token_id=llm_tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 6. Stream with Thinking Visualization
    print("Bot:", end=" ")
    full_response = ""
    is_thinking = False

    for new_text in streamer:
        if "<|im_end|>" in new_text: continue

        full_response += new_text

        # Start of thought
        if "<think>" in new_text:
            is_thinking = True
            print(Fore.YELLOW + "\n(Thinking... ", end="", flush=True)
            new_text = new_text.replace("<think>", "")

        # End of thought
        if "</think>" in new_text:
            is_thinking = False
            new_text = new_text.replace("</think>", "")
            # Print the closing parenthesis manually, then reset color
            print(Fore.YELLOW + new_text + ")\n" + Style.RESET_ALL, end="", flush=True)
            continue  # Skip the standard print so we don't print it twice

        # Standard Print
        if is_thinking:
            # Print thoughts in Yellow
            print(Fore.YELLOW + new_text, end="", flush=True)
        else:
            # Print answer in Default color
            print(Style.RESET_ALL + new_text, end="", flush=True)

    print()  # Newline

    # 7. Clean response for history (Save only the answer, not the thought)
    # This prevents the context window from filling up with old reasoning
    clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

    chat_history.add("user", user_input)
    chat_history.add("you", clean_response)