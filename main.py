from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)


# react_prompt_with_format_instructions = PromptTemplate(
#     template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
#     input_variables=["input", "agent_scratchpad", "tool_names"],
# ).partial(format_instructions=output_parser.get_format_instructions())

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")


tools = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4o-mini")

structured_llm = llm.with_structured_output(AgentResponse)

react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse)

# chain = agent_executor | extract_output | parse_output

chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from langchain-course!")
    # chain = agent_executor
    result = chain.invoke(
        input={"input": "search and give me the top 3 packages in bayesian ML"}
    )

    print(result)


if __name__ == "__main__":
    main()
