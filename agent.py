import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
import json

async def main():
    load_dotenv()

    # Load MCP config from mcp.json
    with open('mcp.json', 'r') as f:
        config = json.load(f)
    
    client = MCPClient.from_dict(config)

    llm = ChatOpenAI("gpt-4o-mini")

    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    # Define the agent's goal
    goal = "Return information about u/alexandroslekkas on Reddit"

    # Run the agent
    result = await agent.run(goal)

    print(result)

if __name__ == "__main__":
    asyncio.run(main())
