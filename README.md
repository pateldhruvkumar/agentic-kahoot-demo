# CrewAI Kahoot Bot

A persistent AI agent that automatically plays Kahoot quizzes using CrewAI Flows and Browser MCP.

## Features

- **Persistent Agent**: Stays connected throughout entire game sessions
- **Event-Driven Architecture**: Uses CrewAI Flow with `@start()` and `@listen()` decorators  
- **Browser Automation**: Leverages Browser MCP for DOM interaction
- **Fast Response**: Aims to answer within 2 seconds using cached knowledge + LLM reasoning
- **Multi-Agent Collaboration**: Navigator, Parser, Knowledge Guru, and Clicker agents work together

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CrewAI Flow   │    │   Browser MCP   │    │   OpenRouter    │
│   Orchestrator  │◄──►│     Server      │◄──►│      LLM        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│     Agents      │    │    Kahoot.it    │
│ • Navigator     │    │   (Persistent   │
│ • Parser        │    │    Session)     │
│ • Knowledge     │    └─────────────────┘
│ • Clicker       │
└─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini

# Browser MCP Server Configuration  
BROWSER_MCP_URL=ws://localhost:8080
BROWSER_MCP_TRANSPORT=websocket

# Kahoot Game Configuration
KAHOOT_PIN=435949
KAHOOT_NICKNAME=CrewAI_Bot
```

### 3. Start Browser MCP Server

```bash
# Install browser MCP server (if not already installed)
npm install -g @browser-mcp/server

# Start the server
browser-mcp-server --port 8080
```

### 4. Run the Kahoot Bot

```bash
python kahoot_bot.py
```

## How It Works

### Flow Architecture

The bot uses **CrewAI Flow** (not Sequential or Hierarchical) because:

- **Event-driven**: Reacts to DOM changes and timer updates
- **Parallel processing**: Can overlap LLM calls with DOM operations  
- **Real-time responsive**: Sub-second reaction times
- **State management**: Maintains game context between questions
- **Conditional routing**: Different paths for cache hits vs misses

### Agent Roles

1. **Navigator**: Joins game, manages browser session, handles reconnects
2. **Parser**: Extracts question text and answer choices from DOM
3. **Knowledge Guru**: Decides best answer (cache → LLM → fallback)
4. **Clicker**: Maps answer to DOM element and executes click

### Flow Execution

```python
@start()
def initialize_bot():
    # Get PIN, join game

@listen(initialize_bot) 
def join_game():
    # Navigate to kahoot.it, enter credentials

@listen(join_game)
def monitor_and_play():
    # Main game loop with question detection
```

## Performance Strategy

- **DOM-only approach**: No screenshots needed, pure text extraction
- **Speed targets**: Complete pipeline in < 2 seconds
- **Cache system**: Store Q→A pairs for instant recall
- **Model selection**: Fast models (gpt-4o-mini) for time pressure
- **Fallback logic**: Heuristic guessing if LLM times out

## Customization

### Different LLM Providers

```env
# Use Anthropic Claude
OPENROUTER_MODEL=anthropic/claude-3-haiku

# Use local Ollama
OPENROUTER_MODEL=ollama/llama3.2:latest
OPENROUTER_API_KEY=ollama
OPENROUTER_BASE_URL=http://localhost:11434/v1
```

### Custom Browser MCP Server

```env
# Different MCP server
BROWSER_MCP_URL=http://localhost:3000/mcp
BROWSER_MCP_TRANSPORT=http
```

## Troubleshooting

### Browser MCP Connection Issues

1. Ensure Browser MCP server is running
2. Check WebSocket/HTTP endpoint is accessible
3. Verify MCP server has browser control permissions

### Game Join Failures

1. Verify Kahoot PIN is active
2. Check network connectivity
3. Ensure browser can access kahoot.it

### Slow Response Times

1. Use faster LLM models (gpt-4o-mini vs gpt-4o)
2. Reduce max_tokens in LLM calls
3. Implement better caching strategy

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Ensure you have permission to use automation tools in your Kahoot games. Respect the terms of service of all platforms used. 