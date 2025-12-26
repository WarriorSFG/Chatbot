import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle

# ================= CONFIGURATION =================
SAMPLES_PER_CLASS = 15000
MAX_FEATURES = 12000
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# ================= LOAD DATA =================
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

FILES = {
    "math": [r"..\Datasets\Train\CoT.jsonl", r"..\Datasets\Train\PoT.jsonl"],
    "code": [r"..\Datasets\Train\Python_Code.jsonl", r"..\Datasets\Train\Python_Syntax.jsonl"],
    "general": [r"..\Datasets\Train\Reddit.jsonl"]
}
# ================= GENERATE ELEMENTARY MATH =================
print("🧮 Generating Simple Arithmetic...")
simple_math = []
ops = ['+', '-', '*', '/', '=', 'plus', 'minus', 'times', 'divided by']

for _ in range(2000):
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    op = random.choice(ops)

    templates = [
        f"{a} {op} {b}",
        f"what is {a} {op} {b}",
        f"calculate {a} {op} {b}",
        f"solve {a} {op} {b}",
        f"{a} {op} {b} ?"
    ]
    simple_math.append(random.choice(templates))

math_keywords = [
    "maths", "mathematics", "calc", "arithmetic", "algebra", "geometry",
    "what is maths", "help me with maths", "i hate maths", "math tutor"
]
simple_math.extend(math_keywords * 50)

SYNTHETIC_MATH = [
    # --- CALCULUS & ANALYSIS ---
    "calculate the laplace transform", "find the derivative of", "integrate the function",
    "solve the differential equation", "limit as x approaches infinity", "gradient of the curve",
    "partial derivative with respect to x", "chain rule application", "area under the curve",
    "maclaurin series expansion", "taylor series approximation", "fourier transform of signal",
    "riemann sum calculation", "definite integral evaluation", "indefinite integral of",
    "find the critical points", "determine local maxima and minima", "point of inflection",
    "implicit differentiation", "related rates problem", "evaluate the double integral",
    "triple integral in spherical coordinates", "divergence theorem", "stokes theorem",
    "green's theorem application", "line integral along the path", "surface integral calculation",
    "solve the initial value problem", "boundary value problem", "homogeneous differential equation",
    "bernoulli differential equation", "exact differential equation", "integrating factor method",
    "laplacian operator", "jacobian matrix determinant", "hessian matrix calculation",
    "lagrange multipliers method", "optimization problem constraint", "l'hopital's rule limit",
    "convergence of the series", "radius of convergence", "power series representation",
    "calculate the curl of the vector field", "calculate the divergence", "directional derivative",
    "parametric equation of the line", "polar coordinate integration", "cylindrical coordinates volume",
    "mean value theorem", "intermediate value theorem", "rolle's theorem",
    "fundamental theorem of calculus", "integration by parts", "u-substitution method",
    "trigonometric substitution", "improper integral convergence", "simpson's rule approximation",
    "trapezoidal rule error", "euler's method for approximation", "runge-kutta method",

    # --- LINEAR ALGEBRA ---
    "compute the eigenvalue", "calculate the eigenvector", "matrix determinant", "vector cross product",
    "dot product of two vectors", "invert the matrix", "find the inverse matrix",
    "orthogonal projection", "basis vectors of the subspace", "null space of the matrix",
    "column space and row space", "rank of the matrix", "reduced row echelon form",
    "gaussian elimination", "cramer's rule solution", "linear independence check",
    "gram-schmidt process", "orthonormal basis", "diagonalize the matrix",
    "singular value decomposition", "lu decomposition", "cholesky decomposition",
    "qr decomposition", "trace of the matrix", "transpose of the matrix",
    "symmetric matrix properties", "hermitian matrix", "unitary matrix",
    "orthogonal matrix", "projection matrix", "change of basis matrix",
    "linear transformation kernel", "image of linear transformation", "dimension of the vector space",
    "subspace verification", "span of the vectors", "cayley-hamilton theorem",
    "jordan canonical form", "positive definite matrix", "eigenbasis calculation",
    "matrix multiplication rules", "identity matrix", "zero matrix",
    "vector addition and subtraction", "scalar multiplication of vector", "angle between two vectors",
    "magnitude of the vector", "unit vector direction", "coordinate vector representation",
    "matrix exponentiation", "tensor product", "outer product of vectors",

    # --- STATISTICS & PROBABILITY ---
    "calculate standard deviation", "variance of the sample", "probability distribution function",
    "bayes theorem application", "poisson distribution probability", "normal distribution curve",
    "correlation coefficient r", "linear regression slope", "hypothesis testing p-value",
    "null hypothesis rejection", "alternative hypothesis", "confidence interval calculation",
    "z-score formula", "t-test for independent samples", "chi-square test for independence",
    "anova table construction", "expected value of random variable", "covariance between variables",
    "conditional probability", "joint probability density", "cumulative distribution function",
    "probability mass function", "binomial distribution formula", "geometric distribution",
    "exponential distribution", "uniform distribution", "gamma distribution",
    "beta distribution", "student's t-distribution", "f-distribution",
    "central limit theorem", "law of large numbers", "sample mean vs population mean",
    "median and mode calculation", "interquartile range", "box plot interpretation",
    "scatter plot correlation", "coefficient of determination r-squared", "least squares method",
    "maximum likelihood estimation", "margin of error", "sample size determination",
    "random sampling method", "stratified sampling", "cluster sampling",
    "permutation and combination", "factorial calculation", "bayes factor",
    "markov chain transition matrix", "monte carlo simulation", "stochastic process",
    "random walk probability", "regression residuals analysis", "homoscedasticity check",

    # --- ALGEBRA & NUMBER THEORY ---
    "solve for x", "quadratic formula roots", "roots of the polynomial",
    "factor the polynomial", "simplify the expression", "expand the binomial",
    "complete the square", "solve the system of equations", "inequality solution set",
    "absolute value equation", "logarithmic equation", "exponential growth decay",
    "compound interest formula", "geometric sequence sum", "arithmetic progression term",
    "complex number arithmetic", "imaginary unit i", "conjugate of complex number",
    "polar form of complex number", "de moivre's theorem", "euler's formula",
    "modulo arithmetic", "greatest common divisor gcd", "least common multiple lcm",
    "prime factorization", "euclidean algorithm", "diophantine equation",
    "congruence relation", "fermat's little theorem", "wilson's theorem",
    "chinese remainder theorem", "primitive root modulo n", "totient function phi",
    "rationalize the denominator", "partial fraction decomposition", "synthetic division",
    "polynomial long division", "remainder theorem", "factor theorem",
    "discriminant of quadratic", "vertex of the parabola", "axis of symmetry",
    "asymptotes of rational function", "domain and range of function", "inverse function",
    "composition of functions", "one-to-one function", "piecewise function graph",
    "solve for variables x y z", "matrix equation ax=b", "determinant properties",

    # --- GEOMETRY & TRIGONOMETRY ---
    "apply pythagoras theorem", "volume of the sphere", "surface area of the cylinder",
    "circumference of the circle", "area of the triangle", "area of the trapezoid",
    "volume of the cone", "volume of the pyramid", "surface area of the prism",
    "trigonometric identity proof", "sine rule application", "cosine rule for side length",
    "tangent function graph", "secant and cosecant", "cotangent identity",
    "inverse trigonometric function", "arcsin arccos arctan", "unit circle coordinates",
    "radians to degrees conversion", "sector area of circle", "arc length formula",
    "equation of the circle", "equation of the ellipse", "equation of the hyperbola",
    "equation of the parabola", "focus and directrix", "eccentricity of conic section",
    "distance formula between points", "midpoint formula", "slope of the line",
    "parallel and perpendicular lines", "angle bisector theorem", "inscribed angle theorem",
    "central angle theorem", "chord length formula", "tangent to the circle",
    "similar triangles ratio", "congruent triangles sss sas", "vectors in 3d space",
    "cross product for area", "dot product for angle", "coordinate geometry proofs",
    "solid of revolution volume", "surface area of revolution", "polar coordinates graph",
    "spherical triangle area", "law of sines spherical", "law of cosines spherical",

    # --- DISCRETE MATH & LOGIC ---
    "truth table for logic gate", "boolean algebra simplification", "predicate logic quantifier",
    "set theory intersection", "union of sets", "subset and superset",
    "power set calculation", "cartesian product of sets", "relation properties reflexive",
    "symmetric and transitive relations", "equivalence relation class", "partial order set",
    "graph theory nodes edges", "adjacency matrix of graph", "shortest path algorithm",
    "dijkstra's algorithm", "breadth-first search bfs", "depth-first search dfs",
    "eulerian path and circuit", "hamiltonian cycle", "planar graph theorem",
    "graph coloring chromatic number", "tree traversal preorder", "binary search tree",
    "pigeonhole principle", "inclusion-exclusion principle", "recurrence relation solution",
    "generating functions", "combinatorial proof", "mathematical induction proof",
    "proof by contradiction", "proof by contrapositive", "direct proof method",
    "logic gates and or not", "karnaugh map k-map", "finite state machine fsm",
    "regular expression regex", "turing machine tape", "computability theory",
    "big o notation complexity", "time complexity analysis", "space complexity",

    # --- ARITHMETIC & AMBIGUITY FIXERS ---
    "how many are left", "how many do i have", "calculate the total", "sum of the values",
    "what is the remainder", "convert fraction to decimal", "percentage increase",
    "percentage decrease calculation", "ratio and proportion", "unit conversion",
    "metric system conversion", "imperial units conversion", "scientific notation",
    "significant figures rules", "rounding numbers", "estimation of answer",
    "mental math trick", "divisibility rules", "order of operations pemdas",
    "bodmas rule application", "evaluate the expression", "simplify the fraction",
    "mixed number to improper fraction", "lowest common denominator", "cross multiply",
    "solve for the unknown", "word problem solution", "train speed problem",
    "work and time problem", "age problem algebra", "mixture problem solution",
    "profit and loss calculation", "simple interest formula", "discount calculation",
    "tax calculation total cost", "average speed formula", "weighted average",
    "geometric mean", "harmonic mean", "prime number check",
    "fibonacci sequence term", "factorial of a number"
]

SYNTHETIC_CODE = [
    # --- PYTHON ---
    "import pandas as pd", "def main():", "return true", "print(f'hello')",
    "pip install requests", "numpy array shape", "matplotlib plot scatter", "flask app route",
    "django model field", "async await python", "try except block", "lambda function",
    "list comprehension python", "dictionary comprehension", "generator expression",
    "decorator function", "context manager with statement", "class inheritance",
    "super().__init__", "staticmethod decorator", "classmethod decorator",
    "dunder methods __str__", "__init__ method", "read csv pandas", "dataframe groupby",
    "matplotlib pyplot", "seaborn heatmap", "scikit-learn train test split",
    "pytorch tensor", "tensorflow keras model", "beautifulsoup scrape",
    "selenium webdriver", "requests get post", "json load dump", "pickle serialize",
    "os path join", "sys argv", "argparse argument", "logging basicconfig",
    "threading thread", "multiprocessing process", "queue data structure",
    "heapq heappush", "collections counter", "itertools combinations",
    "functools reduce", "math sqrt", "random choice", "datetime now",
    "timedelta object", "regex re match", "string formatting", "f-string python",
    "virtual environment venv", "requirements.txt install", "jupyter notebook cell",
    "pypi package upload", "pytest test case", "unittest testcase",
    "docstring format", "type hinting python", "mypy type check",

    # --- JAVASCRIPT / TYPESCRIPT / NODE ---
    "console.log debug", "document.getelementbyid", "npm install package",
    "react component props", "html div tag", "css background color", "json.stringify",
    "fetch api url", "bootstrap class container", "tailwind css utility",
    "const let var", "arrow function syntax", "async function await",
    "promise resolve reject", "callback function", "event listener click",
    "dom manipulation", "localstorage setitem", "sessionstorage", "cookie set",
    "window location href", "navigator useragent", "xmlhttprequest ajax",
    "axios get request", "express js route", "middleware function",
    "mongoose schema model", "mongodb connection string", "node js fs module",
    "path resolve", "process env variable", "module exports", "require import",
    "typescript interface", "type annotation", "generics typescript",
    "enum typescript", "react usestate hook", "useeffect hook", "usecontext hook",
    "redux store provider", "next js page", "vue js directive", "angular component",
    "jquery selector", "chart js canvas", "three js scene", "websocket connection",
    "socket io emit", "jwt token verify", "bcrypt hash password",

    # --- C++ / C / C# ---
    "c++ include iostream", "std::vector push_back", "std::cout print", "std::cin input",
    "using namespace std", "int main return 0", "class public private",
    "constructor destructor", "pointer reference", "memory allocation new delete",
    "smart pointers unique_ptr", "shared_ptr weak_ptr", "template class typename",
    "operator overloading", "virtual function override", "inheritance polymorphism",
    "struct definition", "enum class", "lambda expression c++", "std::map insert",
    "std::string find", "fstream read write", "cmake build system", "makefile target",
    "gcc clang compiler", "gdb debugger", "segmentation fault debug",
    "c# public static void", "console.writeline", "system.collections.generic",
    "list<string> add", "dictionary<key, value>", "linq query select",
    "async task await", "interface implementation", "abstract class",
    "dependency injection", "entity framework context", "asp.net core controller",
    "wpf xaml binding", "unity monobehaviour", "gameobject transform",
    "c language printf", "scanf input", "malloc free", "header file include",
    "pointer arithmetic", "struct typedef", "preprocessor directive #define",

    # --- JAVA ---
    "public static void main", "system.out.println", "import java.util.*",
    "arraylist add", "hashmap put get", "class extends implements",
    "interface definition", "abstract method", "try catch finally",
    "throw new exception", "synchronized block", "thread run start",
    "runnable interface", "stream api filter map", "lambda expression java",
    "optional class", "maven pom.xml", "gradle build.gradle",
    "spring boot application", "restcontroller requestmapping", "autowired annotation",
    "jpa repository", "hibernate entity", "jdbc connection", "resultset query",
    "servlet request response", "jsp page directive", "junit test assertion",
    "log4j logger", "garbage collection system.gc", "object equals hashcode",

    # --- DATABASE (SQL / NoSQL) ---
    "sql select * from", "inner join table on", "left outer join",
    "group by having", "order by desc", "where clause condition",
    "insert into values", "update set where", "delete from table",
    "create table if not exists", "alter table add column", "drop table cascade",
    "primary key constraint", "foreign key references", "unique index",
    "stored procedure", "trigger before insert", "view creation",
    "transaction commit rollback", "postgresql psql", "mysql workbench",
    "sqlite connect", "mongodb findone", "collection aggregate",
    "redis set get", "cassandra cql", "firebase firestore doc",
    "elasticsearch query", "normalization normal form", "acid properties",

    # --- DEVOPS / TOOLS / TERMINAL ---
    "git commit -m", "git push origin master", "git pull rebase", "git branch checkout",
    "git merge conflict", "docker build -t", "docker run -p", "docker-compose up",
    "kubernetes kubectl get pods", "helm install chart", "aws s3 ls", "ec2 instance start",
    "azure cli az login", "gcloud compute instances", "linux command ls -la",
    "grep search text", "chmod +x file", "chown user:group", "ssh user@host",
    "scp transfer file", "tar -xvf archive", "systemctl status service",
    "journalctl logs", "cron job schedule", "bash script #!/bin/bash",
    "powershell get-childitem", "ci/cd pipeline jenkins", "github actions yaml",
    "terraform init apply", "ansible playbook run", "nginx config file",
    "apache virtualhost", "ssl certificate let's encrypt", "dns propagation",

    # --- GENERAL PROGRAMMING CONCEPTS ---
    "debug this error", "fix the syntax", "runtime error exception",
    "stack overflow error", "memory leak detection", "big o notation complexity",
    "algorithm sorting searching", "binary tree traversal", "graph shortest path",
    "dynamic programming solution", "recursion base case", "iterative solution",
    "object-oriented programming oop", "functional programming paradigm",
    "design pattern singleton", "factory pattern", "observer pattern",
    "solid principles", "dry principle", "kiss principle", "clean code practices",
    "refactoring code", "code review comments", "variable naming convention",
    "ide vscode shortcuts", "intellij idea keymap", "vim commands",
    "regular expression regex", "unicode encoding utf-8", "binary hexadecimal",
    "bitwise operations", "networking tcp udp", "http status codes 404",
    "rest api endpoints", "graphql query mutation", "soap xml request",
    "oauth2 authentication", "jwt token security", "cors policy error",
    "xss vulnerability", "sql injection prevention", "csrf protection"
]
SYNTHETIC_MATH.extend(simple_math)

# Map categories to their synthetic lists
SYNTHETIC_MAP = {
    "math": SYNTHETIC_MATH,
    "code": SYNTHETIC_CODE
}

print("🚜 Loading Data...")
data_frames = []

for category, paths in FILES.items():
    for path in paths:
        try:
            df_cat = pd.read_json(path, lines=True)
            df_cat['category'] = category
            df_cat = df_cat.sample(n=min(len(df_cat), SAMPLES_PER_CLASS // len(paths)), random_state=42)
            data_frames.append(df_cat)
        except ValueError:
            continue

print("💉 Injecting Synthetic Data...")
for category, phrases in SYNTHETIC_MAP.items():

    # Create DataFrame
    df_synth = pd.DataFrame({'input': phrases})
    df_synth['category'] = category

    df_synth = pd.concat([df_synth] * 200, ignore_index=True)
    data_frames.append(df_synth)
    print(f"   Injected {len(df_synth)} synthetic rows for '{category}'")

df = pd.concat(data_frames, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print('Constructed dataframe:')
print(df.info())

# ================= CLEAN & VECTORIZE =================
print("🧹 Cleaning & Vectorizing...")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-z0-9+=*<> -]', ' ', text.lower())
    words = text.split()
    return ' '.join([ps.stem(w) for w in words if w not in stop_words])
df['clean_text'] = df['input'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 3))
X_numpy = vectorizer.fit_transform(df['clean_text']).toarray()

encoder = LabelEncoder()
Y_numpy = encoder.fit_transform(df['category'])

with open(r"..\VectorStorage\vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
with open(r"..\VectorStorage\encoder.pkl", "wb") as f: pickle.dump(encoder, f)

# ================= PREPARE TENSORS =================
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
Y_tensor = torch.tensor(Y_numpy, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================= DEFINE MODEL =================
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU to training.")
else:
    print("Using CPU for training.")
model = Net(MAX_FEATURES, 3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= TRAIN LOOP =================
print(f"🚀 Training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

# ================= 6. SAVE MODEL =================
torch.save(model.state_dict(), r"..\Models\Classifier\Router.pth")
print("\n🎉 Training Complete. Model & Vectorizer saved.")