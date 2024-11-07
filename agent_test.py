from custom_agents.chat_planner_agent import chat_planner_agent
from transformers import AutoTokenizer
import os
from langchain_groq import ChatGroq

import google.generativeai as genai

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")



def get_llm():

    api_key = os.environ['GROQ_API_KEY']
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-70b-versatile")

    return llm

def get_long_ctx_llm():

    api_key = os.environ['GEMINI_API_KEY']
    llm = genai.configure(api_key=api_key)

    return llm

def get_chat_expert(_llm, _tokenizer, _planning_llm, _long_ctx_llm):
    print("#45")
    return chat_planner_agent(_llm,_tokenizer, _planning_llm, _long_ctx_llm, log_level='DEBUG', log_file='./agentlogs/chat_planner.txt', logging_enabled=True  )

print("#01")
llm = get_llm()
print("#02")
long_ctx_llm = get_long_ctx_llm()
print("#03")
planning_llm = llm
print("#04")
chat_expert = get_chat_expert(llm,tokenizer, planning_llm, long_ctx_llm)
print("#05")
chat_history = [{"role":"assistant","content":"Hello, ask me something."}]
print("#06")
message = "cuantos son los signos del zodiaco?"
print("#07")
chat_history.append({"role":"user","content":message})
print("#08")
expert_answer = chat_expert.ask_question({"messages": chat_history})
print("#09")
chat_history.append({"role":"assistant","content":expert_answer})

for message in chat_history:
    print(message['role'], message['content'])
print("#10")
