from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    ("system", "you are a {domain} advisor"),
    ("human","a person earns {amount} as a monthly come, and his monthly spend is {spend}, give me single line comment on this")
]

prompt_template = ChatPromptTemplate.from_messages(messages)


chain = prompt_template | model | StrOutputParser()

result = chain.invoke({
    "domain":"finance",
    "amount":100000,
    "spend":80000
})

print(result)

