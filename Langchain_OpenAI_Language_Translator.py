#set environment variables
import os

#Get access key
import Access_Tokens as at

#langchain tools
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage #, AIMessage

#correcting display of arabic characters
import arabic_reshaper
from bidi.algorithm import get_display

#set access env variables 
os.environ["OPENAI_API_TYPE"] = at.OPENAI_API_TYPE
os.environ["OPENAI_API_KEY"] = at.OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = at.OPENAI_API_BASE
os.environ["OPENAI_API_VERSION"] = at.OPENAI_API_VERSION

#set input and output language
input_language = "English"
output_language = "Arabic"

#create LLM
# gpt-35-turbo or gpt-4
llm=AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo",temperature=0)

#create system prompt template
#remove second sentence if you are using gpt-4
system_template = (
    """You are a helpful assistant that translates text provided by human from {input_language} to {output_language}. 
    Do not answer the human questions in any language, just translate the input."""
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

#create human/user prompt template
human_template = "{text}"

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#create final prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


while (True):

    try:

        #get input text from user to be translated
        input_text = input(f"{input_language}: ")

        #check if input text is empty
        if input_text =='':
            print("Please enter the text you want to translate.")
            continue
        
        #check if user wants to stop the translation process
        if input_text == "q":
            break
        
        #generate prompt from template
        messages = chat_prompt.format_prompt(
                input_language=input_language, output_language=output_language, text=input_text
            ).to_messages()
        
        #get answer
        answer = llm(messages=messages)
        #print(f"answer: {answer}\n\n")

        response_message = answer.content

        #format answer (arabic language)
        if output_language == "Arabic":
            response_message = arabic_reshaper.reshape(response_message)
            response_message = get_display(response_message)
        
        #display answer
        print(f"{output_language}: {response_message}")
    
    except Exception as e:
        print(e)

