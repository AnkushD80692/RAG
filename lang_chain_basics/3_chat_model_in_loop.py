from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")

system_message = SystemMessage("You are a finance expert")

chat_history = []
chat_history.append(system_message)

while True:
    query = input("You:")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content = query))
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")