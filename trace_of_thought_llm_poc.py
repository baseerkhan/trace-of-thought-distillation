"""
trace_of_thought_llm_poc.py

Proof-of-Concept for Prompt-Based Knowledge Distillation using Trace-of-Thought prompting.
Author: Baseer Khan | https://www.linkedin.com/in/baseer-khan-656549/
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load student LLM (small and CPU-friendly)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

def teacher_trace_of_thought(question: str):
    """Hardcoded Trace-of-Thought decomposition by teacher."""
    if "Natalia" in question:
        return [
            "Identify April’s sales: 48 clips.",
            "Calculate May’s sales as half of April: 24 clips.",
            "Add April and May sales: 48 + 24 = 72 clips."
        ]
    elif "circle" in question.lower():
        return [
            "Identify the diameter: 10 cm.",
            "Calculate radius: 10 / 2 = 5 cm.",
            "Apply area formula: π * r^2 = π * 25.",
            "Final answer: 25π cm² (~78.5 cm²)."
        ]
    else:
        return [
            "Step 1: Understand the problem.",
            "Step 2: Break into subproblems.",
            "Step 3: Solve and combine answers."
        ]

def generate_student_response(prompt: str, max_tokens=100):
    """Use Hugging Face LLM to complete the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

def build_prompt(question, steps):
    """Combine question and steps into Trace-of-Thought prompt."""
    steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    prompt = (
        f"We are given the following problem:\n{question}\n\n"
        f"Use the following steps to solve it:\n{steps_text}\n\n"
        "Answer:"
    )
    return prompt

if __name__ == "__main__":
    print("=== Prompt-Based Knowledge Distillation ===")
    question = input("Enter your math word problem: ")

    # Teacher Phase
    steps = teacher_trace_of_thought(question)
    print("\n👨‍🏫 Teacher's Steps (Trace-of-Thought):")
    for step in steps:
        print(f"• {step}")

    # Build prompt for student
    prompt = build_prompt(question, steps)
    print("\n📜 Student Prompt:\n", prompt)

    # Student Model Phase
    print("\n🧑‍🎓 Student is generating response...")
    result = generate_student_response(prompt)
    print("\n✅ Student's Output:\n", result)

