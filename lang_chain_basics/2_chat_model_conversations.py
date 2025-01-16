from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")

messages = [
    SystemMessage("You are a finance expert"),
    HumanMessage("Give a short defination of finance eligibility")
]

result = llm.invoke(messages)

print(result.content)
