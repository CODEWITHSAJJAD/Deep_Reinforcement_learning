
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = 't5-small'  # or 'google/flan-t5-large'
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load SQuAD dataset
squad = load_dataset("squad")
train_data = squad["train"]
valid_data = squad["validation"]

def preprocess_function(examples):
    # Prepare input by concatenating question and context
    inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding='max_length', return_tensors="tf")
    answers = [answer['text'][0] if answer['text'] else "" for answer in examples['answers']]
    labels = tokenizer(answers, max_length=64, truncation=True, padding='max_length', return_tensors="tf")
    labels_input_ids = labels["input_ids"]
    labels_input_ids = tf.where(labels_input_ids == tokenizer.pad_token_id, -100, labels_input_ids)
    model_inputs["labels"] = labels_input_ids
    return model_inputs
# Apply preprocessing
train_data = train_data.map(preprocess_function, batched=True)
valid_data = valid_data.map(preprocess_function, batched=True)
# Convert to TensorFlow datasets
train_dataset = train_data.to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    batch_size=8,
)
valid_dataset = valid_data.to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    batch_size=8,
)
# Freeze all layers
for layer in model.layers:
    layer.trainable = False
# Unfreeze the last layer
model.layers[-1].trainable = True
# Compile with optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss)

# Fine-tune the model
model.fit(train_dataset, validation_data=valid_dataset, epochs=3)

# Testing: generate answer for a sample from validation data
sample = valid_data[0]
input_text = "question: " + sample["question"] + " context: " + sample["context"]
input_ids = tokenizer(input_text, return_tensors="tf").input_ids
# Generate answer
output_ids = model.generate(input_ids, max_length=50)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Question:", sample['question'])
print("Answer:", answer)
##########################################################################
# from tensorflow.python.keras.backend import learning_phase
# from transformers import TFAutoModelForSeq2SeqLM,AutoTokenizer
# # Load the pre-trained Flan-T5-large model and tokenizer
# model_name = 't5-small'#"google/flan-t5-large"
# model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)  # Load model
# tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
# from datasets import load_dataset
#
# # Load the SQuAD dataset
# squad = load_dataset("squad")
# train_data = squad["train"]
# valid_data = squad["validation"]
#
# def preprocess_function(examples):
#     # Combine the question and context into a single string
#     inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
#     model_inputs = tokenizer(inputs, max_length=64, truncation=True,padding=True, return_tensors="tf")
#     answers=[e['text'][0] for e in examples['answers']]
#     labels = tokenizer(answers, max_length=64,truncation=True, padding=True, return_tensors="tf")
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# # Preprocess the dataset
# train_data = train_data.map(preprocess_function, batched=True)
# valid_data = valid_data.map(preprocess_function, batched=True)
# rain_dataset = train_data.to_tf_dataset(
#     columns=["input_ids", "attention_mask", "labels"],
#     shuffle=True,
#     batch_size=8,
# )
# valid_dataset = valid_data.to_tf_dataset(
#     columns=["input_ids", "attention_mask", "labels"],
#     batch_size=8,
# )
# from tensorflow.keras.optimizers import Adam
# # Freeze all layers by default (encoder, decoder, embedding layers)
# for layer in model.layers:
#     layer.trainable = False
# # Unfreeze only the final task-specific layer
# model.layers[-1].trainable = True
# # Compile the model with the correct Hugging Face loss function for
# optimizer = Adam(learning_rate=3e-5)
# model.compile(optimizer=optimizer, loss=model.hf_compute_loss)
# # Fine-tune the model on the SQuAD dataset
# model.fit(train_data, epochs=3,validation_data=valid_data)
# # Select a sample from the validation set
# sample = valid_data[0]
# # Tokenize the input text
# input_text = "question: " + sample["question"] + " context: " + sample["context"]
# input_ids = tokenizer(input_text, return_tensors="tf").input_ids
#
# # Generate the output (the model's answer)
# output = model.generate(input_ids)
# answer = tokenizer.decode(output[0], skip_special_tokens=True)
#
# print("Question",sample['question'])
# print("Answer:",answer)