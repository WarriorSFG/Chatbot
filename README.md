

<h1>ChatBot Project</h1>

<h2>Overview</h2>
<p>
This project is a custom-trained chatbot designed to support three specialized behaviors:
</p>
<ul>
  <li>Trash Talking (Reddit-style conversation)</li>
  <li>Coding (Python-focused)</li>
  <li>Maths Solving (reasoning and calculation)</li>
</ul>

<p>
The system uses a 7B parameter base model trained on approximately 1 million rows of curated data.
A wrapper-based classification model controls prompt instructions, execution, and safety.
</p>

<hr>

<h2>Training Data</h2>

<p>Total rows: <strong>~1,000,000</strong></p>

<table>
  <tr>
    <th>Data Type</th>
    <th>Rows</th>
    <th>Percentage</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>Reddit</td>
    <td>600k</td>
    <td>60%</td>
    <td>Personality, sarcasm, trash talk</td>
  </tr>
  <tr>
    <td>Python Code (General)</td>
    <td>200k</td>
    <td>20%</td>
    <td>Python syntax and structure</td>
  </tr>
  <tr>
    <td>Math-to-Python (PoT)</td>
    <td>100k</td>
    <td>10%</td>
    <td>Direct math-to-code translation</td>
  </tr>
  <tr>
    <td>Standard Math (CoT)</td>
    <td>100k</td>
    <td>10%</td>
    <td>Logical reasoning and explanations</td>
  </tr>
</table>

<hr>

<h2>Specializations</h2>

<h3>1. Trash Talking</h3>
<ul>
  <li>Uses Reddit comments and posts</li>
  <li>Dynamically adjusts instruction using subreddit context</li>
  <li>Focus on sarcasm, toxicity, and conversational flow</li>
</ul>

<p>This dataset represents the <strong>core personality</strong> of the bot.</p>

<h3>2. Coding</h3>
<p>Training occurs in three stages:</p>
<ul>
  <li>General Python syntax (loops, functions, classes)</li>
  <li>Math-to-code translation (PoT)</li>
  <li>Logic and reasoning via code</li>
</ul>

<h3>3. Maths</h3>

<h4>Chain-of-Thought (CoT)</h4>
<ul>
  <li>Explains logic step-by-step</li>
  <li>No code execution</li>
  <li>Good for understanding, weaker for precision</li>
</ul>

<h4>Program-of-Thought (PoT)</h4>
<ul>
  <li>Outputs Python code only</li>
  <li>No explanation</li>
  <li>High numerical accuracy</li>
</ul>

<pre>
&lt;TOOL_CALL&gt;
print(10 + 5)
&lt;/TOOL_CALL&gt;
</pre>

<hr>

<h2>Datasets Used</h2>

<h3>Coding</h3>
<ul>
  <li>Evol-Code</li>
  <li>iamtarun/python_code_instructions_18k</li>
</ul>

<h3>Maths</h3>
<ul>
  <li>TIGER-Lab/MathInstruct (PoT)</li>
  <li>microsoft/orca-math-word-problems-200k (CoT)</li>
</ul>

<h3>PoT Sources</h3>
<ul>
  <li>mathqa.json</li>
  <li>gsm_gpt4.json</li>
  <li>numglue.json</li>
  <li>MATH_train.json</li>
  <li>aqua_rat_filtered.json</li>
</ul>

<h3>CoT Sources</h3>
<ul>
  <li>aqua_rat.json</li>
  <li>math50k_camel.json</li>
  <li>gsm_rft.json</li>
  <li>MATH_train.json</li>
  <li>gsm_train.json</li>
</ul>

<hr>

<h2>Base Model</h2>

<p>
<strong>cognitivecomputations/dolphin-2.9.2-qwen2-7b</strong>
</p>

<ul>
  <li>7B parameters (fits in 8GB VRAM)</li>
  <li>Strong maths and coding performance</li>
  <li>Minimal guardrails</li>
</ul>

<hr>

<h2>System Architecture</h2>

<h3>Classification Wrapper</h3>
<ul>
  <li>Receives user input</li>
  <li>Identifies intent</li>
  <li>Injects system instructions</li>
  <li>Controls tool execution</li>
</ul>

<h3>Prompt Format</h3>
<pre>
Instruction: You are a Senior Python Developer.
Prompt: Write a function to reverse a list.
</pre>

<hr>

<h2>Math Tool Execution</h2>

<ol>
  <li>Model returns Python code wrapped in &lt;TOOL_CALL&gt;</li>
  <li>Wrapper extracts and executes code</li>
  <li>Result replaces tool call output</li>
</ol>

<h3>Security Restrictions</h3>
<ul>
  <li>Blocked: os</li>
  <li>Blocked: sys</li>
  <li>Blocked: subprocess</li>
</ul>

<hr>

<h2>Subreddit Mapping</h2>

<pre>
SUBREDDIT_MAP = {
  "FINANCE": "wallstreetbets",
  "GAMING": "gaming",
  "RELATIONSHIPS": "relationship_advice",
  "POLITICS": "politics",
  "TECH": "technology",
  "GENERAL": "AskReddit"
}
</pre>

<hr>

<h2>Short-Term Memory</h2>

<pre>
messages = [
  {"role": "system", "content": "You are a rude Reddit user."},
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "What do you want?"},
  {"role": "user", "content": "Tell me a joke."}
]
</pre>

<hr>

<h2>Dynamic Temperature</h2>

<ul>
  <li>Maths & Code: ≤ 0.1</li>
  <li>Chat / Trash Talk: Higher temperature</li>
</ul>

<hr>

<h2>Current Limitations</h2>

<ul>
  <li>Cannot differentiate between “explain” and “solve”</li>
  <li>Safety depends entirely on wrapper logic</li>
</ul>

<hr>

<h2>Status</h2>

<p><strong>Work in Progress</strong></p>

</body>
</html>
