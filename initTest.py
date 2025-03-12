import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from smolagents import LiteLLMModel
from dotenv import load_dotenv

model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    temperature=0.2,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("What is the revenue growth of Apple in 2020?")