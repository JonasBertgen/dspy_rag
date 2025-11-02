#!/usr/bin/env python3
"""
Fast MCP Client with Weather Tool
Responds with "it is sunny" when asked about weather
"""

import asyncio
import json
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]


class FastMCPClient:
    def __init__(self):
        self.tools = [
            Tool(
                name="get_weather",
                description="Get the current weather information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            )
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return available tools in MCP format"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            }
            for tool in self.tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call"""
        if tool_name == "get_weather":
            return "it is sunny"
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get("method")

        if method == "tools/list":
            return {
                "tools": self.list_tools()
            }

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            result = await self.call_tool(tool_name, arguments)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": result
                    }
                ]
            }

        else:
            raise ValueError(f"Unknown method: {method}")


async def main():
    """Example usage of the MCP client"""
    client = FastMCPClient()

    # List available tools
    print("Available Tools:")
    tools_request = {"method": "tools/list"}
    tools_response = await client.handle_request(tools_request)
    print(json.dumps(tools_response, indent=2))
    print()

    # Call the weather tool
    print("Calling get_weather tool:")
    weather_request = {
        "method": "tools/call",
        "params": {
            "name": "get_weather",
            "arguments": {
                "location": "New York"
            }
        }
    }
    weather_response = await client.handle_request(weather_request)
    print(json.dumps(weather_response, indent=2))
    print()
    print(f"Weather result: {weather_response['content'][0]['text']}")


if __name__ == "__main__":
    asyncio.run(main())
