# GPT-2 Text Generation with Gradio UI

This project demonstrates interactive text generation using the GPT-2 language model from the Hugging Face Transformers library. 
It includes a simple script for generating text and a user-friendly web interface built with Gradio.

## Features

- Load and fine-tune GPT-2 (via `transformers` and TensorFlow).
- Generate text based on a user-provided prompt with customizable decoding parameters.
- Interactive web UI using Gradio for real-time text generation.
- Adjustable parameters: max length, temperature, top-k, top-p, repetition penalty.

## Files

- `main.py` — Core script that loads the GPT-2 model and tokenizer, and defines the text generation function.
- `rungradio.py` — Launches a Gradio web interface to interact with the model.

## Requirements

- Python 3.7+
- TensorFlow
- Transformers (`pip install transformers`)
- Gradio (`pip install gradio`)

## Usage

### Run text generation from command line

```bash
python main.py
```

This runs a sample generation with a fixed prompt and prints the output.

### Launch the Gradio web interface

```bash
python rungradio.py
```
This opens a web UI where you can enter a prompt, tweak generation parameters, and generate text interactively.

Use the sliders to adjust parameters:

- Max Length: Maximum tokens to generate

- Temperature: Controls randomness (higher is more random)

- Top-k: Limits next token candidates to the top k choices

- Top-p: Nucleus sampling probability threshold

- Repetition Penalty: Penalizes repeated tokens to reduce repetition

## Example
Input prompt:

```text
Hello GPT How are you?
```
Possible generated output:
```
Hello GPT How are you? I am an AI language model here to help you generate text...
```

## Notes
The model used is "openai-community/gpt2" with TensorFlow.

The Gradio interface supports sharing the app publicly with "share=True".

## License
This project is open-source and free to use. Feel free to contribute or open issues for improvements!



