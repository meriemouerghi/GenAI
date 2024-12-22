
import openai
import Access_Tokens as at

openai.api_type = at.OPENAI_API_TYPE
openai.api_base = at.OPENAI_API_BASE
openai.api_version = at.OPENAI_API_VERSION
openai.api_key = at.OPENAI_API_KEY

user_prompt = """
Who founded EY?
"""

#user_prompt = "What is 123 raised to the power of 2"

response = openai.ChatCompletion.create(
  engine = "gpt-35-turbo",
  messages = [
	{"role":"system","content":"You are an AI assistant."},
	{"role":"user","content":user_prompt}
  ]
)

answer = response['choices'][0]['message']['content']

print(answer)