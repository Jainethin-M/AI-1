from langchain_core.messages import HumanMessage # helps build highlevel ai apps
from langchain_openai import ChatOpenAI # helps build ai apps using openai models
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent # helps build agents that can reason, act, and learn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    model = ChatOpenAI(temperature = 0) # create a chat model with temperature 0 which is not random and more deterministic
    tools = []
    agent_executor = create_react_agent(model, tools)   

    print("Welcome to the React Agent! Type 'exit' to quit.")
    print("I perform calculation for you."    )

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        print("\nAssistant : ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="", flush=True)
        print()

if __name__ == "__main__":
    main()