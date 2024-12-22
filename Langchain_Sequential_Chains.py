#ACCESS
import Access_Tokens as at

import openai

#Built-in libraries
import os

#Langchain
from langchain import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
# from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ["OPENAI_API_TYPE"] = at.OPENAI_API_TYPE
os.environ["OPENAI_API_KEY"] = at.OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = at.OPENAI_API_BASE
os.environ["OPENAI_API_VERSION"] = at.OPENAI_API_VERSION

############ First Chain => generate a company name

template = "You are a naming consultant for new companies. What is a good name for a company that makes {product}?"

first_prompt = PromptTemplate.from_template(template)

#gpt-35-turbo only is supported by AzureOpenAI
#llm=AzureOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo",temperature=0)

# gpt-35-turbo or gpt-4
llm=AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo",temperature=0)

first_chain = LLMChain(llm=llm, prompt=first_prompt)

product = "colorful socks"

first_answer = first_chain.run({'product': product})

print(f"A good name for a company that makes {product} is '{first_answer}'")

############ Second Chain => generate a catchphrase for the company

template = "Write a catch phrase for company {company_name}?"

second_prompt = PromptTemplate.from_template(template)

second_chain = LLMChain(llm=llm, prompt=second_prompt)

# second_answer = second_chain.run({'company_name':first_answer})

# print(f"A good catchphrase for the company {first_answer} that makes {product} is {second_answer}")


############ Overall Chain

overall_chain = SimpleSequentialChain(chains = [first_chain,second_chain])

catchphrase = overall_chain.run(product)

print(f"A good catchphrase for the company '{first_answer}' that makes {product} is {catchphrase}")
