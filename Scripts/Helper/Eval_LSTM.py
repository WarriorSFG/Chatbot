import json
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ================= CONFIGURATION =================
TOKENIZER_PATH = r"..\..\VectorStorage\tokenizer.pkl"
ENCODER_PATH = r"..\..\VectorStorage\encoder.pkl"
MODEL_PATH = r"..\..\Models\Classifier\LSTM_Router.pth"
TEST_DATA_PATH = r"..\..\Datasets\Test\test_router.jsonl"

MAX_SEQ_LENGTH = 50


# ================= 1. RE-DEFINE CLASSES =================

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
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.out = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.out(self.dropout(torch.relu(self.fc(hidden))))


# ================= 2. LOAD ARTIFACTS =================
print("⏳ Loading LSTM Router...")

with open(TOKENIZER_PATH, "rb") as f:
    # This will now work because 'StemmingTokenizer' is defined above
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# Initialize Model
device = torch.device("cpu")
vocab_size = len(tokenizer.word2idx)
model = LSTMRouter(vocab_size, 100, 128, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"✅ Model Loaded! Vocab Size: {vocab_size}")

# ================= 3. RUN TEST =================
print(f"\n🚀 Starting Test Run on {TEST_DATA_PATH}...")
correct = 0
total = 0

try:
    with open(TEST_DATA_PATH, "r") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            text = row['input']
            expected = row['category'].upper()

            # 1. Tokenize
            sequence = tokenizer.encode([text], MAX_SEQ_LENGTH)

            # 2. Convert to Tensor
            tensor_input = torch.tensor(sequence, dtype=torch.long).to(device)

            # 3. Predict
            with torch.no_grad():
                out = model(tensor_input)
                pred_idx = torch.argmax(out, dim=1).item()
                pred_label = encoder.inverse_transform([pred_idx])[0].upper()

            # 4. Score
            total += 1
            if pred_label == expected:
                correct += 1
                print(f"✅ [PASS] {text[:30]:<30} -> {pred_label}")
            else:
                print(f"❌ [FAIL] {text[:30]:<30} -> Pred: {pred_label} | Exp: {expected}")

    print(f"\n📊 Final Score: {correct}/{total} ({correct / total:.1%})")

except FileNotFoundError:
    print(f"❌ Error: Could not find test file at {TEST_DATA_PATH}")
except Exception as e:
    print(f"❌ An error occurred: {e}")