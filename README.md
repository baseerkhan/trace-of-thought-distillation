# Trace-of-Thought Distillation 🧠

A minimal Python Proof-of-Concept for **prompt-based knowledge distillation**, using `distilgpt2` and simulated reasoning from a "teacher" via **Trace-of-Thought** prompting.

## 🔍 What It Does

- Uses a fixed "teacher" to break a problem into clear reasoning steps.
- Prompts a small student model (`distilgpt2`) to solve the problem using those steps.
- No training, just smart prompting — demonstrating how reasoning can be transferred.

## 📦 Requirements

```bash
pip install torch transformers

## 🚀 Run the PoC

```bash
python trace_of_thought_llm_poc.py

## Example prompt:

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?


## 📖 Inspiration
Based on the paper Trace-of-Thought Prompting: Investigating Prompt-Based Knowledge Distillation Through Question Decomposition (2025).
