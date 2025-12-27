import torch
import torch.nn as nn
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ================= CONFIGURATION =================
MODEL_PATH = r"..\Models\Classifier\Router.pth"
VECTORIZER_PATH = r"..\VectorStorage\vectorizer.pkl"
ENCODER_PATH = r"..\VectorStorage\encoder.pkl"

MAX_FEATURES = 12000


# ================= 1. DEFINE ARCHITECTURE =================

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.5)

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
model = Net(input_dim=MAX_FEATURES, output_dim=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

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
        output = model(tensor_input)
        prediction_index = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][prediction_index].item()
    label = encoder.inverse_transform([prediction_index])[0]

    return label, confidence


# ================= CLASSIFIER =================
if __name__ == "__main__":
    print("Type 'q' to quit.")
    while True:
        user_input = input("\nInput: ")
        if user_input.lower() == 'q': break

        label, conf = get_prediction(user_input)
        print(f"🤖 Prediction: {label.upper()} ({conf:.1%})")