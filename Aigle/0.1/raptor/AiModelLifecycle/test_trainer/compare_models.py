"""
Compare Original vs Fine-tuned Model Outputs

This script compares responses from the original Qwen2.5-1.5B-Instruct
model and the fine-tuned version.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Configuration - paths inside Docker container
MODEL_DIR = "../tmp/models/"

ORIGINAL_MODEL = MODEL_DIR + "Qwen_Qwen3-4B"
FINETUNED_MODEL = MODEL_DIR + "Qwen_Qwen3-1.7B_qwen3-4b-TW-LLM_1bfb371678494e"


def load_models():
    """Load both original and fine-tuned models"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL,
        trust_remote_code=True
    )

    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL,
        dtype=torch.bfloat16,
        # device_map="auto",
        device_map={"": 0},
        trust_remote_code=True
    )

    print("Loading fine-tuned model...")
    # Model was saved as merged full model, load directly
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL,
        dtype=torch.bfloat16,
        # device_map="auto",
        device_map={"": 0},
        trust_remote_code=True
    )

    print("Models loaded successfully!\n")
    return tokenizer, original_model, finetuned_model


def generate_response(model, tokenizer, prompt, max_new_tokens=2048):
    """Generate response from a model"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(prompt)}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"[Error generating response: {e}]"


def compare_outputs(tokenizer, original_model, finetuned_model, prompt):
    """Compare outputs from both models"""
    print("=" * 60)
    print(f"INPUT: {prompt}")
    print("=" * 60)

    print("\n[ORIGINAL MODEL]:")
    print("-" * 40)
    original_response = generate_response(original_model, tokenizer, prompt)
    print(original_response)

    print("\n[FINE-TUNED MODEL]:")
    print("-" * 40)
    finetuned_response = generate_response(finetuned_model, tokenizer, prompt)
    print(finetuned_response)

    print("\n")
    return original_response, finetuned_response


def main():
    print("=" * 60)
    print("Model Comparison: Original vs Fine-tuned")
    print("=" * 60 + "\n")

    tokenizer, original_model, finetuned_model = load_models()

    # Test prompts
    # test_prompts = [
    #     "What is machine learning?",
    #     "How do I make fried rice?",
    # ]
    test_prompts = []

    print("Running comparison with test prompts...\n")
    for prompt in test_prompts:
        compare_outputs(tokenizer, original_model, finetuned_model, prompt)

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Your prompt: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input:
                compare_outputs(tokenizer, original_model, finetuned_model, user_input)
        except KeyboardInterrupt:
            break

    print("Goodbye!")


if __name__ == "__main__":
    main()
