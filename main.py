from dotenv import load_dotenv



load_dotenv()

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor

tools = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4")

react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = react_prompt
)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)



def main():
    print("Hello from langchain-course!")
    chain = agent_executor
    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for ai engineer in SF"
        }
    )

if __name__ == "__main__":
    main()
