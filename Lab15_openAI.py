import os
import openai
from openai import OpenAI

client = OpenAI(
    api_key="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"
)
messages = [{"role": "system", "content": "You are a helpful assistant"}]
while True:
    msg=input("User :")
    if msg:
        messages.append({"role":"user","content":msg})
        chat = client.chat.completions.create(model="gpt-4o-mini",store=True, messages=messages)
    reply=chat.choices[0].message.content
    print("ChatGPT:",reply)
    messages.append({"role": "assistant", "content": reply})

# import openai
# openai.api_key= "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"
# messages=[{"role":"system", "content":"you are a helpful assistant"}]
# while True:
#     msg=input("User : ")
#     if msg:
#         messages.append({"role":"user", "content":msg})
#         chat=openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
#     reply=chat.choices[0].message.content
#     print("chatGPT:",reply)
#     messages.append({"role":"assistant","content":reply})