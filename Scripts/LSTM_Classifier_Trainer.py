import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import os
import re
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ================= CONFIGURATION =================
MAX_VOCAB_SIZE = 12000
MAX_SEQ_LENGTH = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.002
SAMPLES_PER_CLASS = 150000

# ================= 1. TOKENIZER =================
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
        all_words = []
        for text in texts:
            all_words.extend(self.clean(text).split())
        counts = Counter(all_words)
        vocab = ["<PAD>", "<UNK>"] + [w for w, c in counts.most_common(self.max_words - 2)]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"Vocab Size: {len(self.word2idx)}")

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

# ================= 2. GENERATORS =================
def generate_math_sentence(keyword):
    templates = [
        f"how do i calculate the {keyword}?",
        f"what is the formula for {keyword}?",
        f"help me solve this {keyword} problem",
        f"find the {keyword}",
        f"solve for {keyword}",
        f"calculate {keyword}",
        f"evaluate this {keyword}",
        f"i have a homework question about {keyword}",
        f"use {keyword} to solve the equation",
        f"apply the {keyword} rule"
    ]
    if random.random() < 0.1: return keyword
    return random.choice(templates)

def generate_code_sentence(keyword):
    templates = [
        f"write a python script to {keyword}",
        f"how do i implement {keyword}?",
        f"show me an example of {keyword}",
        f"debug this {keyword} issue",
        f"why is {keyword} not working?",
        f"fix the syntax in my {keyword}",
        f"how to install {keyword}?",
        f"import {keyword} library",
        f"run the {keyword} command"
    ]
    if random.random() < 0.1: return keyword
    return random.choice(templates)

SYNTHETIC_MATH = [
    "5 + 5", "10 divided by 2", "square root", "percentage", "remainder",
    "derivative", "integral", "laplace transform", "eigenvalue", "matrix determinant",
    "standard deviation", "variance", "hypothesis testing", "pythagoras theorem",
    "sine rule", "cosine rule", "boolean algebra", "solve for x", "find y"
]

SYNTHETIC_CODE = [
    "import pandas", "def main", "return true", "print(f'hello')",
    "pip install", "numpy array", "matplotlib plot", "console.log",
    "document.getelementbyid", "npm install", "std::cout", "public static void",
    "select * from", "git commit", "docker build", "sudo apt-get",
    "debug this", "fix the syntax", "stack overflow"
]

FILES = {
    "math": [r"..\Datasets\Train\CoT.jsonl", r"..\Datasets\Train\PoT.jsonl"],
    "code": [r"..\Datasets\Train\Python_Code.jsonl", r"..\Datasets\Train\Python_Syntax.jsonl"],
    "general": [r"..\Datasets\Train\Reddit.jsonl"]
}

# ================= 3. BALANCED DATA LOADING =================
print("🚜 Loading Data...")

# A. LOAD REAL DATA
# -----------------
real_dfs = []
for category, paths in FILES.items():
    for path in paths:
        if not os.path.exists(path): continue
        try:
            df_cat = pd.read_json(path, lines=True)
            col = next((c for c in ['input', 'question', 'body'] if c in df_cat.columns), None)
            if col:
                df_cat = df_cat[[col]].rename(columns={col: 'text'})
                df_cat['category'] = category
                real_dfs.append(df_cat)
        except: continue

df_real = pd.concat(real_dfs, ignore_index=True) if real_dfs else pd.DataFrame(columns=['text', 'category'])
print(f"   Loaded {len(df_real)} real samples.")

# B. GENERATE SYNTHETIC DATA
# --------------------------
print("🧪 Generating Synthetic Candidates...")
synth_data = []

# Math: Raw Arithmetic (3000 rows)
ops = ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided by', '%']
for _ in range(3000):
    a = random.randint(1, 1000)
    b = random.randint(1, 1000)
    op = random.choice(ops)
    synth_data.append({'text': f"{a}{op}{b}", 'category': 'math'})
    synth_data.append({'text': f"{a} {op} {b}", 'category': 'math'})

# Math: Concepts
for keyword in SYNTHETIC_MATH:
    for _ in range(20): # Generate plenty, we will balance later
        synth_data.append({'text': generate_math_sentence(keyword), 'category': 'math'})
        synth_data.append({'text': keyword, 'category': 'math'})

# Code: Common Keywords
sql_js = ["SELECT * FROM", "console.log", "import", "def function", "public static void"]
for phrase in sql_js:
    for _ in range(50):
        synth_data.append({'text': phrase, 'category': 'code'})

# Code: Concepts
for keyword in SYNTHETIC_CODE:
    for _ in range(20):
        synth_data.append({'text': generate_code_sentence(keyword), 'category': 'code'})
        synth_data.append({'text': keyword, 'category': 'code'})

df_synth = pd.DataFrame(synth_data)

# C. MERGE & FORCE BALANCE
# ------------------------
print("⚖️  Balancing Dataset...")
df_all = pd.concat([df_real, df_synth], ignore_index=True)

# Separate by category
df_math = df_all[df_all['category'] == 'math']
df_code = df_all[df_all['category'] == 'code']
df_general = df_all[df_all['category'] == 'general']

# Function to force exact count
def balance_df(df, target_n):
    if len(df) == 0: return df # Avoid crash if empty
    if len(df) >= target_n:
        return df.sample(n=target_n, random_state=42) # Downsample
    else:
        return df.sample(n=target_n, replace=True, random_state=42) # Upsample

# Balance each
df_math_bal = balance_df(df_math, SAMPLES_PER_CLASS)
df_code_bal = balance_df(df_code, SAMPLES_PER_CLASS)
df_general_bal = balance_df(df_general, SAMPLES_PER_CLASS)

# Combine & Shuffle
df = pd.concat([df_math_bal, df_code_bal, df_general_bal], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("-" * 30)
print(f"📊 Final Distribution (Should be {SAMPLES_PER_CLASS} each):")
print(df['category'].value_counts())
print("-" * 30)

# ================= 4. TRAINING =================
print("🔢 Tokenizing (with Stemming)...")
tokenizer = StemmingTokenizer(MAX_VOCAB_SIZE)
tokenizer.fit(df['text'].values)

X_numpy = tokenizer.encode(df['text'].values, MAX_SEQ_LENGTH)
encoder = LabelEncoder()
Y_numpy = encoder.fit_transform(df['category'])

# Save Artifacts
os.makedirs(r"..\VectorStorage", exist_ok=True)
with open(r"..\VectorStorage\tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)
with open(r"..\VectorStorage\encoder.pkl", "wb") as f: pickle.dump(encoder, f)

# LSTM Model
class LSTMRouter(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMRouter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.out = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.out(self.dropout(torch.relu(self.fc(hidden))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRouter(len(tokenizer.word2idx), EMBEDDING_DIM, HIDDEN_DIM, 3).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

X_train, X_test, y_train, y_test = train_test_split(X_numpy, Y_numpy, test_size=0.1)
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True)

print(f"🚀 Training Balanced LSTM on {device}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == by).sum().item()
        total += by.size(0)

    print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Acc: {correct / total:.2%}")

os.makedirs(r"..\Models\Classifier", exist_ok=True)
torch.save(model.state_dict(), r"..\Models\Classifier\LSTM_Router.pth")
print("🎉 Done.")