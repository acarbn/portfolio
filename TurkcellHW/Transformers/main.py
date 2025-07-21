from datasets import load_dataset
from text_processor import process_text
from transformers import Trainer, TrainingArguments
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq

import torch
from rouge_score import rouge_scorer
import numpy as np


# Load the CNN/DailyMail summarization dataset
data = load_dataset("cnn_dailymail", "3.0.0",split="train[:5000]")  # 3.0.0 is the latest version (uses corrected stories)
datav=load_dataset("cnn_dailymail", "3.0.0",split="validation[:100]")
datat=load_dataset("cnn_dailymail", "3.0.0",split="test[:100]")

"""data = load_dataset("cnn_dailymail", "3.0.0",split="train")  # 3.0.0 is the latest version (uses corrected stories)
datav=load_dataset("cnn_dailymail", "3.0.0",split="validation")
datat=load_dataset("cnn_dailymail", "3.0.0",split="test")"""

texts=data["article"] 
labels=data["highlights"] 

textsv=datav["article"] 
labelsv=datav["highlights"] 

textst=datat["article"] 
labelst=datat["highlights"] 

#print(labels[:50])
print(len(texts))
print(len(labels))
#print(texts)
labels = [str(label) for label in labels]  # This converts everything to string
labelst = [str(label) for label in labelst] 
labelsv = [str(label) for label in labelsv] 
clean_texts=[process_text(text) for text in texts]
#clean_labels=[process_text(label) for label in labels]
clean_labels=labels
clean_textsv=[process_text(text) for text in textsv]
#clean_labelsv=[process_text(label) for label in labelsv]
clean_labelsv=labelsv

clean_textst=[process_text(text) for text in textst]
#clean_labelst=[process_text(label) for label in labelst]
clean_labelst=labelst

# Tokenizer ve Model


model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


max_input_length = 512 # max enabled by t-5 small 
max_target_length = 150 

def tokenize_data(texts, labels):
    inputs = ["summarize: " + text for text in texts]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length", return_tensors="pt")
    labelstok = tokenizer(labels, max_length=max_target_length, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labelstok["input_ids"]
    return model_inputs

train_encodings = tokenize_data(clean_texts, clean_labels)
val_encodings = tokenize_data(clean_textsv, clean_labelsv)
test_encodings=tokenize_data(clean_textst, clean_labelst)

#print(train_encodings)
#print(val_encodings)

# Pytorch Dataset
class SummaryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

train_dataset = SummaryDataset(train_encodings)
val_dataset = SummaryDataset(val_encodings)
test_dataset= SummaryDataset(test_encodings)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert to numpy just in case
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Now compute ROUGE-L
    rouge_l = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(decoded_labels, decoded_preds)
    ]

    return {"rougeL": np.mean(rouge_l)}



class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": max_target_length,
            "num_beams": 4,
            "return_dict_in_generate": True,
            "output_scores": False,
        }

        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            ).sequences

            labels = inputs["labels"]

            # Pad generated tokens if necessary
            if generated_tokens.shape[1] < labels.shape[1]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, labels.shape[1])
            elif labels.shape[1] < generated_tokens.shape[1]:
                labels = self._pad_tensors_to_max_len(labels, generated_tokens.shape[1])

        return (None, generated_tokens, labels)


    def _pad_tensors_to_max_len(self, tensor, max_length):
        pad_token_id = tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=torch.long).to(tensor.device)
        padded_tensor[:, :tensor.shape[1]] = tensor
        return padded_tensor


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,  # try 6 first
    learning_rate=3e-5,  # much better default for BART
    warmup_steps=500,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

save_directory = "news_summary/saved_model_bart_trainer_epoch6_noclearlabel"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Evaluation on test set...")
predictions = trainer.predict(test_dataset)
metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print(f"\n📊 Mean ROUGE-L (batch predict): {metrics['rougeL']:.4f}")

decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

for i in range(3):
    print(f"\n--- Example {i+1} ---")
    print(f"Reference: {decoded_labels[i]}")
    print(f"Generated: {decoded_preds[i]}")

