"""Helper functions for the workshop notebooks"""
from langchain.pydantic_v1 import BaseModel

def simple_description_formatter(tool:BaseModel):
    """outputs a json description of the tool."""
    properties = {k:{'type':tool.args[k]['type'], 'description':tool.args[k]['description']}  for k in tool.args.keys() }

    return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tool.args.keys()),
                },
            },
        }