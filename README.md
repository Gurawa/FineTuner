# Fine-tuning T5 for Psychology Responses

This project fine-tunes the T5-small model to generate helpful responses to psychological concerns. The dataset used contains instructions, input texts, and corresponding output texts.

## Description

The goal of this project is to fine-tune the T5-small model to provide helpful responses to various psychological concerns. The model is trained on a dataset with examples of concerns and appropriate responses.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Fine-tuning the Model

To fine-tune the T5-small model, run the `finetune_t5.py` script:

```bash
python finetune_t5.py
```

This script will load the dataset, fine-tune the model, and save the fine-tuned model in the `./finetuned_t5` directory.

### Running the Fine-tuned Model

To run the fine-tuned model and generate responses for new inputs, run the `run_finetuned_t5.py` script:

```bash
python run_finetuned_t5.py
```

Example usage:

```python
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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
