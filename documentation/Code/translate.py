import json
from transformers import MarianMTModel, MarianTokenizer

# 1. Load Translation Model (English → Irish)

model_name = "Helsinki-NLP/opus-mt-en-ga"  # English to Irish
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 2. Load JSONL Data
input_file = "data/massive_amr.jsonl"
output_file = "data/massive_amr_irish.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    
# 3. Translate Function
def translate_text(text, tokenizer, model):
    if not text.strip():
        return text  # Skip empty strings

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return translated_text

# 4. Translate Sentences
for entry in data:
    entry["utt"] = translate_text(entry["utt"], tokenizer, model)
    entry["annot_utt"] = translate_text(entry["annot_utt"], tokenizer, model)

# 5. Save Translated Data
with open(output_file, "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
print("Translation complete! Saved as 'massive_amr_irish.jsonl'")