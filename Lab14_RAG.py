import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
# Sample documents
documents = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is a historic fortification.",
    "Python is a popular programming language."
]
def encode(texts):
    inputs=tokenizer(texts,padding=True,truncation=True,return_tensors='tf')
    with tf.GradientTape() as tape:
        outputs=model.get_encoder()(**inputs)
        embeddings=tf.reduce_mean(outputs.last_hidden_state,axis=1)
    return embeddings.numpy()
############################################
doc_embeddings=encode(documents)
def retrieve(query,doc_embeddings,documents):
    query_emb=encode([query])[0]
    sims=np.array([np.dot(query_emb,doc_emb)/(np.linalg.norm(query_emb)*np.linalg.norm(doc_emb))for doc_emb in doc_embeddings])
    best_idx=np.argmax(sims)
    return documents[best_idx],sims[best_idx]
#######################################
question="Where is Eiffel tower"
retrived_doc,sim_score=retrieve(question,doc_embeddings,documents)
input_text="question: "+question+" context: "+retrived_doc
inputs=tokenizer(input_text,return_tensors='tf',padding=True)
outputs=model.generate(**inputs,max_length=50)
answer=tokenizer.decode(outputs[0],skip_special_tokens=True)
print("Question:",question)
print("Retrieved answer=",retrived_doc)
print("LLM answer=",answer)