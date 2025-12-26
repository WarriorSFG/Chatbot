import json
import torch
import torch.nn as nn
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# SETUP
ps = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'([+=*<>/-])', r' \1 ', text)  # Space out operators
    text = re.sub(r'[^a-z0-9+=*<> -]', ' ', text)
    words = text.split()
    return ' '.join([ps.stem(w) for w in words if w not in stop_words])


# LOAD MODEL
print("⏳ Loading Router...")
with open(r"..\..\VectorStorage\vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
with open(r"..\..\VectorStorage\encoder.pkl", "rb") as f: encoder = pickle.load(f)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net(len(vectorizer.get_feature_names_out()), 3)
model.load_state_dict(torch.load(r"..\..\Models\Classifier\Router.pth"))
model.eval()

# RUN TEST
print("\n🚀 Starting Test Run...")
correct = 0
total = 0

with open(r"..\..\Datasets\Test\test_router.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        text = row['input']
        expected = row['category'].upper()

        # Predict
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned]).toarray()
        tensor = torch.tensor(vec, dtype=torch.float32)

        with torch.no_grad():
            out = model(tensor)
            pred_idx = torch.argmax(out).item()
            pred_label = encoder.inverse_transform([pred_idx])[0].upper()

        total += 1
        if pred_label == expected:
            correct += 1
            print(f"✅ [PASS] {text[:30]:<30} -> {pred_label}")
        else:
            print(f"❌ [FAIL] {text[:30]:<30} -> Pred: {pred_label} | Exp: {expected}")

print(f"\n📊 Final Score: {correct}/{total} ({correct / total:.1%})")