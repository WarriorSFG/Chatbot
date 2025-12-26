import re
import torch
import pickle
import torch.nn as nn
from threading import Thread
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer

MODEL_PATH = r"Models\Dominator_Dolphin"
SYSTEM_PROMPT = """"""

# ================= CONFIGURATION =================
CLASSIFIER_PATH = r"Models\Classifier\Router.pth"
VECTORIZER_PATH = r"VectorStorage\vectorizer.pkl"
ENCODER_PATH = r"VectorStorage\encoder.pkl"

MAX_FEATURES = 12000


# ================= DEFINE ARCHITECTURE =================
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ================= LOAD ARTIFACTS =================
print("⏳ Loading model components...")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

device = torch.device("cpu")
Classifier = Net(input_dim=MAX_FEATURES, output_dim=3)
Classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
Classifier.eval()

print("✅ Model Loaded!")

# ================= PREPROCESSING LOGIC =================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-z0-9+=*<> -]', ' ', text.lower())
    words = text.split()
    return ' '.join([ps.stem(w) for w in words if w not in stop_words])


# ================= PREDICT FUNCTION =================
def get_prediction(text):
    # 1. Clean
    cleaned_text = clean_text(text)
    # 2. Vectorize
    vectorized_input = vectorizer.transform([cleaned_text]).toarray()
    # 3. Convert to Tensor
    tensor_input = torch.tensor(vectorized_input, dtype=torch.float32).to(device)
    # 4. Predict
    with torch.no_grad():
        output = Classifier(tensor_input)
        prediction_index = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][prediction_index].item()
    label = encoder.inverse_transform([prediction_index])[0]

    return label, confidence


# ================= PROMPTS =================
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

def CODE_PROMPT():
    instruction = "You are a Senior Python Developer. Write a Python program to solve this."
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
        label, conf = get_prediction(user_input)
        print(f"🤖 Prediction: {label.upper()} ({conf:.1%})")
    except EOFError:
        break

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Bye.")
        break

    '''
    is_technical = any(
        word in user_input.lower() for word in ['code', 'python', 'script', 'solve', 'calculate', 'math', 'function'])
    current_temp = 0.1 if is_technical else 1.8'''

    #Set Temperature
    if label == 'MATH' or label == 'CODE':
        current_temp = 0.1
    else:
        current_temp = 0.8

    content = REDDIT_PROMPT(subreddit='RoastMe')
    if label == 'MATH':
        content = MATH_PROMPT(Code=True, LaTeX=False, Explain=False, Comments=False)
    elif label == 'CODE':
        content = CODE_PROMPT()

    messages = [
        {"role": "system", "content": content},
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
