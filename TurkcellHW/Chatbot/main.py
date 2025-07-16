# Fine-tuning: train the model with our data
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf

model_name="openai-community/gpt2"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=TFAutoModelForCausalLM.from_pretrained(model_name)

text="Hello GPT How are you?"
encoded_input=tokenizer(text,return_tensors="tf")
#output=model(encoded_input)
#print(output)
output=model.generate(**encoded_input,max_length=100,
                      pad_token_id=tokenizer.eos_token_id,
                      do_sample=True,
                      temperature=0.8,
                      top_k=100,
                      top_p=0.5,
                      repetition_penalty=1.2
                      ) # top possible
# Generate text
#generated_ids = model.generate(**encoded_input, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

def generate_text(prompt, max_length=100, temperature=0.8, top_k=100, top_p=0.5):
    encoded_input = tokenizer(prompt, return_tensors="tf")
    output = model.generate(
        **encoded_input,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
