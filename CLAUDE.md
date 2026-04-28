# MCP Server for Lego Robot

It controls a 4 motor lego robot connected to a Raspberry Pi via the BuildHat HAT and OV5647 Pi Camera.

## Tool use and Code synthesis
1. The agentic coder should prefer using the MCP server to control the robot, as it should be more reliable and consistent.
2. If no appropriate function or tool is missing, the agentic coder should modify the MCP server code itself while testing. For instance, if you want the robot to grasp something, and the function does not exist, you should add the function to the MCP server code and test it.
3. Inspect the logs of the MCP server for debugging. The logs are available at `mcp_robot/logs/mcp_server.log`.

## Technical Details

1. The MCP runs in virtual environment.
2. The MCP server hot-reloads on code changes.
3. The MCP server uses persistent SSH connection to the RPi via `paramiko`.