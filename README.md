# Bill Processing MCP Server

A MCP-based server for processing and analyzing bill documents (PDFs and images)

## Features

- PDF and image bill parsing
- Text extraction from documents
- Structured bill data extraction
- Bill querying capability
- General chat functionality
- Bill data export
- Bill management (listing, retrieval)

## Prerequisites

- Python 3.8+
- uv package manager
- OpenAI API key

## Installation

1. Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies using uv:
```bash
uv pip install mcp[cli]
uv install
```

## Configuration

1. Set your OpenAI API key:
```bash
set OPENAI_API_KEY=your-api-key-here  # Windows
# or
export OPENAI_API_KEY=your-api-key-here  # Unix/Linux
```
Note: This is not required when used with Claude desktop

## Usage

1. Test the MCP server:
```bash
uv run --with mcp[cli] mcp run server.py
```

2. The server provides the following tools:

- `parse_pdf_file`: Extract data from PDF bills
- `parse_bill_image`: Extract data from bill images
- `query_bill`: Ask questions about processed bills
- `general_chat`: General conversation capability
- `export_bill_data`: Export bill data to JSON
- `get_bill_by_id`: Retrieve specific bill data
- `list_all_bills`: List all processed bills

3. Install the server for Claude-Desktop use:
```bash
mcp install server.py
```

The Claude Desktop configuration should look like this:

```json
{
  "mcpServers": {
    "Bill Processing MCP": {
      "command": "C:\\Users\\user\\Desktop\\projects\\THT_Talentica\\.venv\\Scripts\\uv.exe",
      "args": [
        "run",
        "--with",
        "asyncio",
        "--with",
        "mcp[cli]",
        "--with",
        "openai",
        "--with",
        "pydantic",
        "--with",
        "pymupdf",
        "mcp",
        "run",
        "C:\\Users\\user\\Desktop\\projects\\THT_Talentica\\server.py"
      ]
    }
  }
}
```

## Project Structure

```
THT_Talentica/
├── .venv/
├── logs/
│   └── document_management.log
├── exported_bills/
├── server.py
├── requirements.txt
└── project.yaml
```

## Error Handling

- The server includes comprehensive error handling and logging
- Logs are stored in `logs/document_management.log`
- Failed operations return appropriate error messages

## Notes

- Processed bills are stored in memory during runtime
- Exported bills are saved in the `exported_bills` directory

