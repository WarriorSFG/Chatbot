import re

import nltk
import torch
import pickle
import torch.nn as nn
import numpy as np
from threading import Thread

from nltk import PorterStemmer
from nltk.corpus import stopwords
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer

MODEL_PATH = r"Models\Dominator_Dolphin"

# ================= CONFIGURATION =================
# Update these to point to your new LSTM files
CLASSIFIER_PATH = r"Models\Classifier\LSTM_Router.pth"
TOKENIZER_PATH = r"VectorStorage\tokenizer.pkl"
ENCODER_PATH = r"VectorStorage\encoder.pkl"

MAX_SEQ_LENGTH = 50  # Must match your training script


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
        pass  # Not needed for inference

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        # Space out operators
        text = re.sub(r'([+=*<>/-])', r' \1 ', text)
        text = re.sub(r'[^a-z0-9+=*<> -]', ' ', text)
        words = text.split()
        return ' '.join([self.ps.stem(w) for w in words if w not in self.stop_words])

    def encode(self, texts, max_len):
        tokenized = []
        for text in texts:
            words = self.clean(text).split()
            # Use 1 for <UNK>
            ids = [self.word2idx.get(w, 1) for w in words[:max_len]]
            # Pad with 0
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            tokenized.append(ids)
        return np.array(tokenized)

class LSTMRouter(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMRouter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.out = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concat forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.out(self.dropout(torch.relu(self.fc(hidden))))


# ================= 2. LOAD ARTIFACTS =================
print("⏳ Loading Classifier Components...")

# Load Tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load Encoder
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# Load Model
device = torch.device("cpu")  # LSTM is small enough for CPU inference
# We need to reconstruct the model with the same dimensions used in training
# vocab_size is derived from the loaded tokenizer
classifier_model = LSTMRouter(
    vocab_size=len(tokenizer.word2idx),
    embed_dim=100,
    hidden_dim=128,
    output_dim=3
)
classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier_model.eval()

print("✅ Classifier Loaded!")


# ================= 3. NEW PREDICT FUNCTION =================
def get_prediction(text):
    # 1. Tokenize (Note: encode expects a list)
    sequence = tokenizer.encode([text], MAX_SEQ_LENGTH)

    # 2. Convert to Tensor (Long type for Embedding layer)
    tensor_input = torch.tensor(sequence, dtype=torch.long).to(device)

    # 3. Predict
    with torch.no_grad():
        output = classifier_model(tensor_input)

        # Get probabilities
        probs = torch.softmax(output, dim=1)
        prediction_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction_index].item()

    # 4. Decode Label
    label = encoder.inverse_transform([prediction_index])[0]
    return label, confidence


# ================= 4. LLM SETUP & PROMPTS =================
print("⏳ Loading Dominator Dolphin...")
model, llm_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0"
)
FastLanguageModel.for_inference(model)

if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

terminators = [
    llm_tokenizer.eos_token_id,
    llm_tokenizer.convert_tokens_to_ids("<|im_end|>")
]


# ================= PROMPTS =================
def REDDIT_PROMPT(subreddit=None, context=None):
    # Default to AskReddit if no subreddit is provided
    sub_name = subreddit if subreddit else "AskReddit"

    # 1. Define a stronger, more creative persona
    base_instruction = (
        f"You are a cynical, witty, and savage redditor on r/{sub_name}. "
        "Your goal is to roast the user with creative sarcasm and specific mockery. "
        "RULES:\n"
        "- DO NOT repeat yourself.\n"
        "- Use the context provided to find specific things to make fun of.\n"
        "- Keep it short, punchy, and informal."
    )

    # 2. Append context if it exists
    if context:
        return f"{base_instruction}\n\n--- CONVERSATION CONTEXT ---\n{context}\n\nNow reply to the user:"

    return f"{base_instruction}\n\nReply to the user:"

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

def CODE_PROMPT():
    instruction = "You are a Senior Python Developer. Write a Python program to solve this."
    return instruction

# --- HISTORY CLASS ---
class History:
    def __init__(self, tokenizer, max_seq_len=2048, response_reserve=512):
        self.history = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.response_reserve = response_reserve

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_safe_context(self, current_input, system_prompt):
        """
        Returns a string of history that fits within the remaining token limit.
        """
        # 1. Calculate the Fixed Costs (System + Input + Reserve)
        # We assume strict formatting "User: ... \n" adds minimal overhead
        sys_tokens = len(self.tokenizer.encode(system_prompt))
        input_tokens = len(self.tokenizer.encode(current_input))

        # 2. Calculate Available Budget for History
        available_budget = self.max_seq_len - (sys_tokens + input_tokens + self.response_reserve)

        if available_budget <= 0:
            print("⚠️ Warning: Input is too long! Dropping all history.")
            return ""

        # 3. Iterate Backwards (Newest -> Oldest)
        selected_messages = []
        current_cost = 0

        for msg in reversed(self.history):
            # Format exactly how it appears in the prompt
            msg_str = f"{msg['role'].title()}: {msg['content']}\n"
            msg_cost = len(self.tokenizer.encode(msg_str))

            if current_cost + msg_cost <= available_budget:
                selected_messages.append(msg_str)
                current_cost += msg_cost
            else:
                # Stop immediately if the next message breaks the budget
                break

                # 4. Reverse back to Chronological Order (Oldest -> Newest) and join
        return "".join(reversed(selected_messages))

chat_history = History(llm_tokenizer, max_seq_len=2048, response_reserve=512)

# ================= 5. MAIN LOOP =================
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

    # 1. PREDICT INTENT (Using LSTM)
    label, conf = get_prediction(user_input)
    label = label.upper().strip()
    print(f"🤖 Prediction: {label} ({conf:.1%})")

    # 2. CONFIGURATION
    if label == 'MATH':
        base_prompt = MATH_PROMPT(Code=False)
        current_temp = 0.2
    elif label == 'CODE':
        base_prompt = CODE_PROMPT()
        current_temp = 0.2
    else:
        base_prompt = REDDIT_PROMPT(subreddit='RoastMe')
        current_temp = 0.8

    print(f"DEBUG: Temp set to {current_temp}")

    # 3. BUILD PROMPT
    context_str = chat_history.get_safe_context(user_input, base_prompt)

    if label == 'MATH':
        content = MATH_PROMPT(Code=False)
    elif label == 'CODE':
        content = CODE_PROMPT()
    else:
        content = REDDIT_PROMPT(subreddit='RoastMe', context=context_str)

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": user_input},
    ]

    # 4. GENERATE
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        streamer=streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=current_temp,
        min_p=0.1,
        eos_token_id=terminators,
        pad_token_id=llm_tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Bot:", end=" ")
    full_response = ""
    for new_text in streamer:
        if "<|im_end|>" not in new_text:
            print(new_text, end="", flush=True)
            full_response += new_text
    print()

    chat_history.add("user", user_input)
    chat_history.add("bot", full_response)