# Fine-tuning a BERT Model on the GLUE MRPC dataset using Hugging Face Transformers.

Step:

Install dependencies
```
pip install transformers datasets
```

Info:

* FineTuner.py loads dataset from huggingface.
* Tokenizes the text from the data.
* Sets up training argument for fine-tuning.
* Saves the model.
