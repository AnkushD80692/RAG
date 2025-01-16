from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")

template="a person earns {amount} as a monthly come, and his monthly spend is {spend}, give me single line comment on this"
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({
    "amount":100000,
    "spend":80000
})

result = llm.invoke(prompt)
print(result.content)

messages = [
    ("system", "you are a {domain} advisor"),
    ("human","a person earns {amount} as a monthly come, and his monthly spend is {spend}, give me single line comment on this")
]
prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({
    "domain":"finance",
    "amount":100000,
    "spend":80000
})

result = llm.invoke(prompt)
print(result.content)





