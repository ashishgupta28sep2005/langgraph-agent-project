# LangGraph Agent Demo

An isolated, small runnable application inside the LangGraph monorepo. The default mock mode needs no API key and demonstrates graph routing with a local arithmetic tool.

## Windows PowerShell

```powershell
cd agent-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python agent.py
python -m pytest
```

Python 3.11 or later is required; Python 3.12 works. In mock mode, try `What is 12 * 3?` and then `Hello`.

For console OpenAI mode, set `AGENT_MODE=openai` and `OPENAI_API_KEY` in the process environment before running `python agent.py`. For CLI mode, copy `.env.example` to `.env` and set the key there; the CLI loads that file. Secrets are not committed. The optional model defaults to `gpt-4o-mini` and can be changed with `OPENAI_MODEL`.

The included `langgraph.json` exports `agent:graph` for LangGraph CLI use. Install the CLI with `python -m pip install 'langgraph-cli[inmem]'`, then run `langgraph dev` from this directory. The local API is at `http://127.0.0.1:2024`; the API is optional because the console app runs directly with Python.

The graph starts at `plan`. Mock mode routes basic arithmetic through the `calculator` node, then ends at `respond`; other prompts take the direct mock response route. OpenAI mode binds the same calculator tool to the configured model, which can request the tool before returning its final answer.
