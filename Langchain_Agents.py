#ACCESS
import Access_Tokens as at

import openai

#Built-in libraries
import os

#Langchain
from langchain import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
# from langchain.llms import AzureOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, get_all_tool_names

os.environ["OPENAI_API_TYPE"] = at.OPENAI_API_TYPE
os.environ["OPENAI_API_KEY"] = at.OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = at.OPENAI_API_BASE
os.environ["OPENAI_API_VERSION"] = at.OPENAI_API_VERSION

############ First Chain => generate a company name

# prompt = """When was the 3rd president of the United States born? 
# What is that year raised to the power of 3?"""

#Human tool
#prompt = "What's the 3rd character in my name?"

#wikipedia
#prompt = "When was the 3rd president of the United States born?"


prompt = """How much money will I have if I spend 30$?"""
#"""When was I born? What is that year raised to the power of 3?"""

#gpt-35-turbo only is supported by AzureOpenAI
#llm=AzureOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo",temperature=0)

# gpt-35-turbo or gpt-4
llm=AzureChatOpenAI(deployment_name="gpt-4", model_name="gpt-4",temperature=0
                    )

tools = load_tools(["llm-math","human","google-search"], llm=llm)
print(get_all_tool_names())

agent = initialize_agent(tools, llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
                        )
try:
    answer = agent.run(prompt)
except Exception as e:
    answer = str(e)
finally:
    l = answer.split("\n")
    #print(l[-1])
    print(answer)
    # final_answer = [line for line in l if line.startswith("Final Answer")]
    # print(final_answer[0])