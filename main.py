from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

from geminiAIlist import list_gemini_models


def main():
    # Gemini model (choose one that exists in your account, common: "gemini-1.5-flash")
    model = ChatGoogleGenerativeAI(
    # model="gemini-2.5-pro", error
    # model="gemini-flash-latest", # working
    model="gemini-2.5-flash",
    temperature=0,
)


    tools = [list_gemini_models]
    # tools=[]
    agent_executor = create_react_agent(model, tools)

    print("Welcome to the React Agent! Type 'exit' to quit Gemini.")
    print("I perform calculation for you.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("\nAssistant : ", end="")
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    if message.content:
                        print(message.content, end="", flush=True)
        print()

if __name__ == "__main__":
    main()
