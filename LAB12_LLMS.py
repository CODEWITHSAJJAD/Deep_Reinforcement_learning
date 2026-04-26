from transformers import TFAutoModelForSeq2SeqLM,AutoTokenizer
model_name='t5-small'
model=TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizers=AutoTokenizer.from_pretrained(model_name)
from datasets import load_dataset
squad=load_dataset('squad')
train_data=squad['train']
valid_data=squad['validation']
#########################################################
def preprocess_function(examples):
    inputs = ['question: ' + q + ' context: ' + c for q, c in zip(examples['question'], examples['context'])]
    model_inputs = tokenizers(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='tf')
    return model_inputs
#########################################################
train_data=train_data.map(preprocess_function,batched=True)
valid_data=valid_data.map(preprocess_function,batched=True)
import tensorflow as tf
for layer in model.layers:
    layer.trainable=False
model.layers[-1].trainable=True
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss=model.hf_compute_loss)
model.fit(train_data.shuffle(1000).batch(8),epochs=3,validation_data=valid_data.batch(8))
sample=valid_data[0]
input_text=('question: '+sample['question']+'context: '+sample['context'])
input_ids=tokenizers(input_text,return_tensors='tf')
output=model.generate(input_ids)
answer=tokenizers.decode(output[0],skip_spacial_tokens=True)
print("Question=",sample['question'])
print("Context=",sample['context'])
print("Answer=",answer)
######################################################
# from seaborn import load_dataset
# from transformers import TFAotoModelForSeq2SeqLM, AutoTokenizer
#
# from ANNDL.BIRD import tokenizer
#
# model_name = 'google/flan-t5-large'
# model = TFAotoModelForSeq2SeqLM.from_pretrained(model_name)
# Tokenizer = AutoTokenizer.from_pretrained(model_name)
# squad = load_dataset('squad')
# train_data = squad['train']
# valid_data = squad['validation']
#
#
# ###################################################################
# def preprocess_function(examples):
#     inputs = ['question: ' + g + "context: " + c for q, c in zip(examples['question'], examples['context'])]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensor='tf')
#     return model_inputs
#
#
# train_data = train_data.map(preprocess_function, batch=True)
# valid_data_data = valid_data.map(preprocess_function, batch=True)
# from tensorflow.keras.optimizers import Adam
#
# for layer in model.layers:
#     layer.trainable = False
# model.layers[-1].trainable = True
# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss=model.hf_compute_loss)
# model.fit(train_data.shuffle(1000).batch(8), epochs=3, validation_data=valid_data.batch(8))
# sample = valid_data[0]
# input_text = 'question:' + sample['question'] + 'context'
# input_ids = tokenizer(input_text, return_tensors='tf')
# output = model.generate(input_ids)
# answer = tokenizer.decode(output[0], skip_special_tokens=True)
# print("Questions = ", sample['question'])
# print("Answer=", answer)
###########       TASK     ###########################
# from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
# model_name = 'google/flan-t5-large'
# model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizers = AutoTokenizer.from_pretrained(model_name)
# from datasets import load_dataset
# squad = load_dataset('squad')
# train_data = squad['train']
# valid_data = squad['validation']
# #########################################################
# def preprocess_function(examples):
#     inputs = ['question: ' + q + ' context: ' + c for q, c in zip(examples['question'], examples['context'])]
#     model_inputs = tokenizers(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='tf')
#     return model_inputs
# #########################################################
# train_data = train_data.map(preprocess_function, batched=True)
# valid_data = valid_data.map(preprocess_function, batched=True)
# from tensorflow.keras.optimizers import Adam
# for layer in model.layers:
#     layer.trainable = False
# model.layers[-1].trainable = True
# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss=model.hf_compute_loss)
# model.fit(train_data.shuffle(1000).batch(8), epochs=3, validation_data=valid_data.batch(8))
# user_question = input("Please enter your question: ")
# user_context = input("Please enter the context: ")
# input_text = f'question: {user_question} context: {user_context}'
# input_ids = tokenizers(input_text, return_tensors='tf')
# output = model.generate(input_ids)
# answer = tokenizers.decode(output[0], skip_special_tokens=True)
# print("Question =", user_question)
# print("Context =", user_context)
# print("Answer =", answer)
