from importlib.metadata import version
from pathlib import Path
from sympy import simplify
import torch
import re
import json
import requests
import time
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError

from ch02 import (get_device, generate_text_basic_stream)

from reasoning_from_scratch.qwen3 import(
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)

from IPython.display import Latex, display

used_libraries = [
    "reasoning_from_scratch",
    "torch",
    "sympy",
    "tokenizers"  # Used by reasoning_from_scratch
]

""" for lib in used_libraries:
    print(f"{lib} version: {version(lib)}") """


""" 
Math problem solving has become a cornerstone of evaluating and developing reasoning models
 Our step-by-step math verifier will follow
 1 - Load pretrained LLM
 2 - Ask LLM to generate answer
 3 - Extract final answer box
 4 - Normalize extracted answer
 5 - Verify equivalence
 6 - Grade answers
 7 - Load MATH-500 dataset
 8 - Evaluate the model

Below we are loading the "base" model
to load the reasoning version we change as follows:
WHICH_MODEL = "reasoning" 
 """
WHICH_MODEL = "base"
device = get_device()
""" Regular expression for extracting numeric values from text  """
RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

""" Latex formatting to be replaced -left is original value, right is new value """
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

""" strip chat special tokens like <|assistant|> """
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # 

""" Dictionary mapping to convert unicode superscripts to plaintext superscript """
SUPERSCRIPT_MAP = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
    }

def load_model_and_tokenizer(
    which_model, device, use_compile, local_dir="qwen3"
):
    if which_model == "base":
        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir)/"qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    elif which_model == "reasoning":
        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )
    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    return model, tokenizer

#print(device)

model, tokenizer = load_model_and_tokenizer(
    which_model=WHICH_MODEL,
    device=device,
    use_compile=False
)

prompt = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)

""" 
Below is exactly as we did in Ch02 on line 185
 """
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
).unsqueeze(0)

""" 
Moved code below into generate_text_stream_concat()
all_token_ids = []

for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0)
    decoded_id = tokenizer.decode(token_id.tolist())
    print(
        decoded_id,
        end="",
        flush=True
    )
    all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids)

 """

""" 
With the promptabove the model produces the following

See notes
The model provides reasoning-model-like  explanations because
the Qwen3 team included chain-of-thought data during the pretraining as 
stated in their technical report

The final answer returned is in the form
"\boxed{\\dfrac{14}{3}}"
boxed format answeres are a common convention in math benchmarks and training.
it reflects that pretrained models often encounter many problem formats online
and learn to reproduce  those stylistic conventions
 """

#display(Latex(all_tokens))


# This is stp 2 in our 8 steps
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens, verbose=False
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)
    generated_ids = []
    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)

""" 
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens=2048, verbose=False
)
 """

model_answer = (
r"""... some explanation...
**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
""") # this is the answer box we want to extract Step 3 in our 8 steps

def get_last_boxed(text):
    """ find last occurence of "\boxed" """
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None
    """ get position after "\boxed" """
    current_idx = boxed_start_idx + len(r"\boxed")
    
    """ skip any whitespace after "\boxed" """
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth >0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1
    if brace_depth != 0:
        return None
    return text[content_start_idx:current_idx-1]


""" 
extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)

 """

""" 
Start 06-27-2026

The above extracted the text but we can also make extraction more robust 
The following accounts for instances where the final answer box is missing or incomplete (3.5)
 """

def extract_final_candidate(text, fallback="number_then_full"):
    """ default value if there is no match """
    result = ""
    if text:
        """ prefer the last boxed expression if present """
        boxed = get_last_boxed(text.strip())
    if boxed:
        result = boxed.strip().strip("$ ")
        """ if no boxed expression try fallback """
    elif fallback in ("number_then_full", "number_only"):
        """ Below we are using  RE_NUMBER defined on line 52, which is regex 
        designed to recognize numbers, including fractions, decimal and scientific notation
        """
        m = RE_NUMBER.findall(text)
        if m:
            """ use last numner """
            result = m[-1]
        elif fallback == "number_then_full":
            """ return full text if no number is found """
            result = text
    return result
""" 

print(extract_final_candidate(r"\boxed{ 14/3. }"))
print(extract_final_candidate("abc < > 14/3 abc"))
 """

""" 
Our goal now becomes to normalize the response into a more general
canonical form.
Models may return the same response in many ways, so we need a robust 
checking system to determine the accuracy of the response
To this we added LATEX_FIXES, RE_SPECIAL and SUPERSCRIPT_MAP after the regex RE_NUMBER
(3.6) - all uppercase defines have been included at the top of the file
The function below takes an extracted answer string and rewrites it into a standardized format
that can be reliably compared against reference solutions
"""

def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    """ Strip leading multiple-choice labels
    E.g., like "c. 3" -> 3, or "b: 2" -> 2 """
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
    """ Remove angle-dgree markers like ^{\\circ}, ^\\circ and unicode degree respectively """
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")

    """ Unwrap "\text{}" if the whole string is wrapped """
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")
    """ strip inline/display math wrappers \\( \\) \\[ \\] """
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    """ LaTex canonicalization """
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)
    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"
    """ convert unicode superscripts into exponent form (e.g., 2² -> 2**2)m """
    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)
    """ Normalize number and root expressions """
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    """ Convert LaTex fractions into division form """
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    """ Handle exponents and mixed numbers """
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )
    """ Remove thousands separators in numbers 1,234 -> 1234 """
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )
    return text.replace("{", "").replace("}", "").strip().lower()


#print(normalize_text(extract_final_candidate(model_answer)))
#print(normalize_text(r"$\dfrac{14}{3.}$"))
#print(normalize_text(r"\text{\[\frac{14}{3}\]}"))
#print(normalize_text("4/3"))


""" 
Our next goal is to verify the extracted answer by comparing it to a correct reference, called
the "ground truth"
Note that for this we cannot use a Python's == operator since "14/3" and "(14)/(3)" would not match
We'll parse the extracted normalized answer with a synmbolic math engine: https:/ /sympy.org (3.7)
 """
def sympy_parser(expr):
    """ Done to avoid crashing on long garbage responses from badly trained models """
    if expr is None or len(expr) > 2000:
        return None;
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # standard transformations like handling parentheses
                *spp.standard_transformations,
                # allow omitted multiplication symbols (e.g., "2x" -> 2*x")
                spp.implicit_multiplication_application,
            ),
            # Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError, 
        IndexError, TokenError, ValueError, PolynomialError):
        return None

""" 
print(sympy_parser(normalize_text(
    extract_final_candidate(model_answer)
)))
print(sympy_parser("28/6"))
 """

""" 
With the sympy_parser we can no implement the equality checker (3.8)
 """
def equality_check(expre_gtruth, expr_pred):
    """ Check if the two expressions are equal """
    if expre_gtruth == expr_pred:
        return True
    """ parse both expressions into SynPy objects (returns None if parsing fails) """
    gtruth, pred = sympy_parser(expre_gtruth), sympy_parser(expr_pred)
    """ if both expressions were parsed successfully we try symbolic comparison """
    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0
        except(SympifyError, TypeError):
            pass
    return False


""" print(equality_check(
    normalize_text("13/4."),
    normalize_text(r"(13)/(4)")
)) """

""" print(equality_check(
    normalize_text("0.5"),
    normalize_text(r"(1)/(2)")
)) """

""" print(equality_check(
    normalize_text("14/3"),
    normalize_text("15/3")
)) """

""" print(equality_check(
    normalize_text("(14/3, 2/3)"),
    normalize_text("(14/3, 4/6)")
)) """
# returns False since model cannot handle tuple-like expressions

""" 
Here we create a helper function to split tuple-like expressions (3.9)
if the answer is tuple-like (a, b) or [a, b] it splits the content on the commas
and returns the individual pieces
 """
def split_into_parts(text):
    result = [text]
    if text:
        if(
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        result = []
    return result

""" 
print(split_into_parts(normalize_text(r"(14/3, 2/3)")))
# split helper returns: ['14/3', '2/3'] as expected

 """

""" 
Below we create the grade_answer function with can take tuple-like results
split them with the (3.9) helper funtion and then use the (3.8) function 
to compare the answers
the grade_answer below is an advanced implementation of equality_check above
3.10
 """
def grade_answer(pred_text, gt_text):
    result = False
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )
        if(
            gt_parts and pred_parts
            and len(gt_parts) == len(pred_parts)
        ):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )
    return result

""" 
print(grade_answer("14/3", r"\frac{14}{3}"))
print(grade_answer(r"(14/3, 2/3)", "(14/3, 4/6)"))

 """

""" 
We define below test cases with name, prediction, ground truth, and txpected results
then we create a function to test them all
3.11
 """
tests = [
        ("check_1", "3/4", r"\frac{3}{4}", True),
        ("check_2", "(3)/(4)", r"3/4", True),
        ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
        ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
        ("check_5", "(1, 2)", r"(1,2)", True),
        ("check_6", "(2, 1)", "(1, 2)", False),
        ("check_7", "(1, 2, 3)", "(1, 2)", False),
        ("check_8", "0.5", "1/2", True),
        ("check_9", "0.3333333333", "1/3", False),
        ("check_10", "1,234/2", "617", True),
        ("check_11", r"\text{2/3}", "2/3", True),
        ("check_12", "50%", "1/2", False),
        ("check_13", r"2\cdot 3/4", "3/2", True),
        ("check_14", r"90^\circ", "90", True),
        ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
        ("check_16", r"2²", "2**2", True),
    ]

def run_demos_table(tests):
    header = ["Test", "Expect", "Got", "Status"]
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))
    data = [header] + rows
    col_widths = [
        max(len(row[i])for row in data)
        for i in range(len(header))
    ]
    for row in data:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)
    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\nPassed {passed}/{len(rows)}")

""" 
run_demos_table(tests)

 """

""" 
We now have a verification pipeline.
Now we want to load the MATH-500 dataset (https:/ /huggingface.co/datasets/HuggingFaceH4/MATH-500)
It is large enough to be meaningful but still small
It is a common benchmark dataset in the reasoning-model literature
(3.12) we imported json and requests for the function below
 """

def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    return data

""" uncomment below to load the math_data """
#math_data = load_math500_test()
#print("Number of entries:", len(math_data))

""" 
We are now in aposition to test the LLM text generation and evaluation tools we've developed above
First, we create a function to increase the likelihood that the return from the LLM is in boxed form
Note: Using no prompt template improves the base model's performance by 50%, but reduces the reasoning models
accuracy by 40%
Note: The standard prompt commonly used for MATH-500 benchmark swaps "Question": for "Problem" - increases base model 
accuracy by 20%, but the reasoning accuracy drops by 30% 
Smaller base models are often quite sensitive to prompt phrasing
(3.13)
 """
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template

""" 
#Code to test render_prompt
prompt = (  # Same prompt we used at the beginning of the chapter
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)
prompt_fmt = render_prompt(prompt)
print(prompt_fmt)
 """

""" 
Let's do and end-to-end test of our pipeline
(3.14)
 """
def mini_eval_model(model, tokenizer, device):
    ex = {
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    """ Apply prompt template """
    prompt = render_prompt(ex["problem"])
    gen_text = generate_text_stream_concat(model, tokenizer, prompt, device,
        max_new_tokens=64
    )
    pred_answer = extract_final_candidate(gen_text)
    is_correct = grade_answer(pred_answer, ex["answer"])
    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")

""" 
mini_eval_model(model, tokenizer, device)
When we run the above we get the following results
Device: cuda
Prediction: 1/3
Ground truth: 2/3
Correct: False
 """

""" 
Helper function to calculate remaining time
(3.15)
 """
def eta_progress_message(
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress"
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)
    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)
    remaining = max(total - processed, 0)
    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0
    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"
    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.json")
    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f: # save results for inspection
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            gen_text = generate_text_stream_concat(
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens, verbose=verbose,
            )
            extracted = extract_final_candidate(gen_text)
            is_correct = grade_answer(extracted, row["answer"])
            num_correct +=int(is_correct)
            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct)
            }
            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:  # Print responses during the generation process
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )
    # Print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc

""" 
print("Model:", WHICH_MODEL)
print("Device:", device)
num_correct, num_examples, acc = evaluate_math500_stream(
    model, tokenizer, device,
    math_data=math_data[:10],#Only evaluate on the first 10 examples.
    max_new_tokens=2048,
    verbose=False
)
 """



""" 
The above will output somthing similar to
Model: base
Device: cuda
MATH-500: 10/10 | ETA: 00s       
Accuracy: 20.0% (2/10)
Total time: 0.4 min
Logs written to: math500-cuda.json
end 06-27-2026 (ToDo - look over the exercises)
 """








