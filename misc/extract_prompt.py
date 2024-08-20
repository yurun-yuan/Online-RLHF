from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

def extact_prompt(sample):
    return {"context_messages": [{"content": sample["prompt"], "role": "user" }]}

dataset = dataset.map(extact_prompt)
dataset.push_to_hub("RyanYr/ultrafeedback_prompts")
