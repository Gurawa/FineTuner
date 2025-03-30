import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./finetuned_t5')
tokenizer = T5Tokenizer.from_pretrained('./finetuned_t5')

# Function to generate responses
def generate_response(input_text, instruction=""):
    # Prepare the input text
    input_text = f"{instruction} {input_text}"
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
instruction = "If you are a licensed psychologist, please provide this patient with a helpful response to their concern."
input_text = "I'm feeling really anxious lately and I don't know why."
response = generate_response(input_text, instruction)
print("Input:", input_text)
print("Response:", response)
