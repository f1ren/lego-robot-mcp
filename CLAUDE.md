# MCP Server for Lego Robot

It controls a 4 motor lego robot connected to a Raspberry Pi via the BuildHat HAT and OV5647 Pi Camera.

The agentic coder should prefer using the MCP server to control the robot. But, if no appropriate function or tool is missing, the agentic coder should modify the MCP server code itself while testing. For instance, if you want the robot to grasp something, and the function does not exist, you should add the function to the MCP server code and test it.

## Technical Details

1. The MCP runs in virtual environment.