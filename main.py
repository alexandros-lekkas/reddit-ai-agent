import asyncio
import os
import time
import json

# MCP Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    MCPSettings,
    MCPServerSettings,
    OpenAISettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Load MCP servers from mcp.json
with open("mcp.json", "r") as f:
    mcp_config = json.load(f)
servers_dict = {}
for name, server in mcp_config.get("mcpServers", {}).items():
    servers_dict[name] = MCPServerSettings(name=name, **server)

settings = Settings(
    execution_engine="asyncio",
    mcp=MCPSettings(servers=servers_dict),
    openai=OpenAISettings(api_key=os.getenv("OPENAI_API_KEY")),
)

# Initialize MCP app
app = MCPApp(name="reddit-ai-agent", settings=settings)

async def example_usage():
    async with app.run() as agent_app:
        context = agent_app.context

        print("Current config:", context.config.model_dump())

        reddit_agent = Agent(
            name="reddit-agent",
            instruction="""You are an agent with access to the Reddit API. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["reddit-mcp-server"],
        )

        async with reddit_agent:
            print("reddit-agent: Connected to server, calling list_tools...")
            result = await reddit_agent.list_tools()
            print("reddit-agent: list_tools result:", result)

            llm = await reddit_agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message="Find informaation about user alexandroslekkas on Reddit"
            )
            print("reddit-agent: generate_str result:", result)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Time taken: {t:.2f} seconds")