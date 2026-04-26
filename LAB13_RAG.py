import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM,AutoTokenizer
import numpy as np
# load t5 model and tokenize
model_name='t5-small'
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
#sample documet
documents=[
    "the Eiffel tower is located in Paris."
    "The Great Wall of China is a historic fortification."
    "Python is a popular programing language"
]
def encode(text):
    inputs=tokenizer(text,padding=True,truncation=True,return_tensors='tf')
    with tf.GradientTape() as tape:
        output=model.get_encoder()(**inputs)
        embedding=tf.reduce_mean(output.last_hidden_state,axis=1)
        return embedding.numpy()
#####################################
doc_embedding=encode(documents)
def retrieve(query,documets_embeddings,documets):
    query_emb=encode([query])[0]
    sims=np.array(([np.dot(query_emb,doc_emb)/(np.linalg.norm(query_emb)*np.linalg.norm(doc_emb))for doc_emb in documets_embeddings]))
    best_idx=np.argmax(sims)
    return documets[best_idx],sims[best_idx]
##############################################
question="where is the Eiffel Tower"
retrieved_doc,sim_score=retrieve(question,doc_embedding,documents)
input_text="question: "+question+" context: "+retrieved_doc
inputs=tokenizer(input_text,return_tensors='tf',padding=True)
outputs=model.generate(**inputs,max_length=50)
answer=tokenizer.decode(outputs[0],skip_special=True)
print("Question=",retrieved_doc)
print("Answer= ",answer)