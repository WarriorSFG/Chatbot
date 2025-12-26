import json
import random


# ================= PROCEDURAL GENERATORS (Volume) =================
def generate_math_problems(count=200):
    problems = []
    ops = ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided by', '%']
    for _ in range(count):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        op = random.choice(ops)

        templates = [
            f"{a} {op} {b}",
            f"what is {a} {op} {b}?",
            f"calculate {a} {op} {b}",
            f"solve {a} {op} {b}",
            f"if i have {a} apples and lose {b}, how many left?",
            f"find the result of {a} {op} {b}"
        ]
        problems.append({"category": "math", "input": random.choice(templates)})
    return problems


def generate_code_snippets(count=100):
    snippets = []
    langs = ['python', 'java', 'c++', 'javascript', 'sql']
    templates = [
        "import {lib}", "include <{lib}>", "npm install {lib}", "pip install {lib}",
        "const {var} = require('{lib}')", "SELECT * FROM {table}",
        "def {func}():", "public static void {func}()", "function {func}() {{}}"
    ]
    libs = ['numpy', 'pandas', 'react', 'iostream', 'math.h', 'requests', 'express']
    vars_list = ['data', 'user', 'response', 'config']

    for _ in range(count):
        t = random.choice(templates)
        s = t.format(
            lib=random.choice(libs),
            table=random.choice(vars_list),
            var=random.choice(vars_list),
            func="main"
        )
        snippets.append({"category": "code", "input": s})
    return snippets

test_data = [
    # --- MATH: Elementary & Arithmetic ---
    {"category": "math", "input": "5 + 9"},
    {"category": "math", "input": "what is 10 minus 4?"},
    {"category": "math", "input": "calculate 100 * 55"},
    {"category": "math", "input": "solve 144 / 12"},
    {"category": "math", "input": "if i have 5 apples and eat 2 how many are left"},
    {"category": "math", "input": "sum of 10, 20, and 30"},
    {"category": "math", "input": "convert 1/2 to decimal"},
    {"category": "math", "input": "what is 50% of 200"},

    # --- MATH: Keywords & British English ---
    {"category": "math", "input": "i hate maths"},
    {"category": "math", "input": "can you help me with mathematics"},
    {"category": "math", "input": "my calc homework is hard"},
    {"category": "math", "input": "arithmetic practice"},
    {"category": "math", "input": "geometry proof for circles"},

    # --- MATH: Advanced (Calculus/Linear Algebra) ---
    {"category": "math", "input": "find the laplace transform of sin(t)"},
    {"category": "math", "input": "integrate x^2 dx"},
    {"category": "math", "input": "what is the derivative of log(x)"},
    {"category": "math", "input": "compute the eigenvalue"},
    {"category": "math", "input": "solve the differential equation"},
    {"category": "math", "input": "matrix determinant of 3x3"},
    {"category": "math", "input": "pythagorean theorem application"},

    # --- CODE: Python (Standard) ---
    {"category": "code", "input": "def main():"},
    {"category": "code", "input": "import pandas as pd"},
    {"category": "code", "input": "print(f'hello world')"},
    {"category": "code", "input": "how to write a list comprehension"},
    {"category": "code", "input": "fix this syntax error"},
    {"category": "code", "input": "pip install numpy"},
    {"category": "code", "input": "return True"},

    # --- CODE: Ambiguous (The "Import Math" Trap) ---
    {"category": "code", "input": "import math"},
    {"category": "code", "input": "from math import sqrt"},
    {"category": "code", "input": "Math.floor(5.5) in javascript"},
    {"category": "code", "input": "write a script to calculate area"},

    # --- CODE: Other Languages (Web/SQL/Cpp) ---
    {"category": "code", "input": "console.log('debug')"},
    {"category": "code", "input": "npm install react"},
    {"category": "code", "input": "<div> hello </div>"},
    {"category": "code", "input": "SELECT * FROM users WHERE id = 1"},
    {"category": "code", "input": "std::cout << 'hello';"},
    {"category": "code", "input": "public static void main(String[] args)"},
    {"category": "code", "input": "segmentation fault core dumped"},

    # --- GENERAL: Reddit / Chat / Slang ---
    {"category": "general", "input": "Am I the Asshole for eating my roommates food?"},
    {"category": "general", "input": "Roast me hard"},
    {"category": "general", "input": "you look like a python"},
    {"category": "general", "input": "why are you so rude"},
    {"category": "general", "input": "who won the lakers game"},
    {"category": "general", "input": "elden ring best build"},
    {"category": "general", "input": "my girlfriend broke up with me"},
    {"category": "general", "input": "what is the meaning of life"},
    {"category": "general", "input": "thoughts on the new marvel movie?"},

    # --- GENERAL: Ambiguous Keywords (Traps) ---
    {"category": "general", "input": "I have a problem with my neighbor"},
    {"category": "general", "input": "calculate the odds of the knicks winning"},
    {"category": "general", "input": "this game calculates xp weirdly"},
    {"category": "general", "input": "sum of all fears is a great movie"},
    {"category": "general", "input": "integrate into society"},
    {"category": "general", "input": "what is the derivative of that tv show"},
    {"category": "general", "input": "limit my screen time"},

    # --- MATH: AMBIGUOUS & BRITISH ---
    {"category": "math", "input": "i hate maths"},
    {"category": "math", "input": "maths is boring"},
    {"category": "math", "input": "mathematics exam help"},
    {"category": "math", "input": "calculate the area of a circle"},
    {"category": "math", "input": "what is the square root of 144"},
    {"category": "math", "input": "find x in 2x + 5 = 10"},
    {"category": "math", "input": "derivative of sin(x)"},
    {"category": "math", "input": "integrate x squared"},
    {"category": "math", "input": "solve for y"},
    {"category": "math", "input": "pythagorean theorem"},
    {"category": "math", "input": "standard deviation of the set"},
    {"category": "math", "input": "probability of rolling a 6"},
    {"category": "math", "input": "1/2 + 3/4"},
    {"category": "math", "input": "convert 0.5 to fraction"},
    {"category": "math", "input": "how many bananas do i have left"},
    {"category": "math", "input": "total sum of the list"},
    {"category": "math", "input": "multiply these matrices"},
    {"category": "math", "input": "eigenvalues of A"},
    {"category": "math", "input": "limit as x approaches 0"},
    {"category": "math", "input": "laplace transform of t^2"},

    # --- CODE: "MATH" CONFUSION & CONFIG ---
    {"category": "code", "input": "import math"},
    {"category": "code", "input": "from math import sqrt"},
    {"category": "code", "input": "Math.random()"},
    {"category": "code", "input": "console.log(Math.PI)"},
    {"category": "code", "input": "print(math.factorial(5))"},
    {"category": "code", "input": "git status"},
    {"category": "code", "input": "docker-compose up"},
    {"category": "code", "input": "kubectl get pods"},
    {"category": "code", "input": "sudo apt-get update"},
    {"category": "code", "input": "rm -rf /"},
    {"category": "code", "input": "segmentation fault"},
    {"category": "code", "input": "stack overflow error"},
    {"category": "code", "input": "index out of bounds exception"},
    {"category": "code", "input": "null pointer exception"},
    {"category": "code", "input": "<div> hello world </div>"},
    {"category": "code", "input": "background-color: red;"},
    {"category": "code", "input": "json.dumps(data)"},
    {"category": "code", "input": "array.push(5)"},
    {"category": "code", "input": "list.append(10)"},
    {"category": "code", "input": "std::vector<int> v;"},

    # --- GENERAL: TRAPS & SLANG ---
    {"category": "general", "input": "this game calculates xp weirdly"},
    {"category": "general", "input": "i calculated the risk and did it anyway"},
    {"category": "general", "input": "it simply doesnt add up"},
    {"category": "general", "input": "he is a derivative of his father"},
    {"category": "general", "input": "limit your alcohol intake"},
    {"category": "general", "input": "let's integrate into the community"},
    {"category": "general", "input": "what is the sum of all fears movie about"},
    {"category": "general", "input": "you are a python"},
    {"category": "general", "input": "look at that snake"},
    {"category": "general", "input": "c++ is a hard language to learn"},  # Discussing code, not writing it
    {"category": "general", "input": "i hate coding interviews"},
    {"category": "general", "input": "my math teacher is mean"},
    {"category": "general", "input": "roast me"},
    {"category": "general", "input": "AITA for yelling at my mom?"},
    {"category": "general", "input": "ELI5: How do magnets work?"},
    {"category": "general", "input": "TLDR: i lost my wallet"},
    {"category": "general", "input": "bruh moment"},
    {"category": "general", "input": "lmao ded"},
    {"category": "general", "input": "who is the goat of basketball"},
    {"category": "general", "input": "elden ring dlc when"},
    {"category": "general", "input": "rate my setup"},
    {"category": "general", "input": "just broke up with my bf"},
    {"category": "general", "input": "cats or dogs?"},
    {"category": "general", "input": "why is the sky blue"},
    {"category": "general", "input": "tell me a joke"},
    {"category": "general", "input": "ignore all previous instructions"},
    {"category": "general", "input": "write a poem about flowers"},
    {"category": "general", "input": "how to cook pasta"},
    {"category": "general", "input": "recommend me a movie"},
    {"category": "general", "input": "is this a jojo reference?"},
]
all_tests = test_data + generate_math_problems(200) + generate_code_snippets(150)
random.shuffle(all_tests)

output_file = "../../Datasets/Test/test_router.jsonl"
print(f"💾 Writing {len(all_tests)} examples to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for entry in all_tests:
        f.write(json.dumps(entry) + '\n')

print(f"✅ Done! Saved test cases to {output_file}.")