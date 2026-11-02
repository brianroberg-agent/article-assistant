# Article Assistant

A Python script to extract metadata from articles and generate Markdown headers for note-taking. Supports The New Atlantis (specialized extraction) and generic websites (LLM-based extraction).

## Features

- **Multi-site support**: Extract metadata from The New Atlantis and generic websites
- **Specialized extraction** for The New Atlantis with no LLM overhead
- **LLM-based extraction** for generic sites using Simon Willison's `llm` utility
- Extracts article metadata including title, author(s), and issue information
- Generates properly formatted YAML front matter for Markdown notes
- Supports both JSON-LD structured data and HTML fallback parsing
- **Automatically infers edition numbers** from season information when explicit numbers aren't available (The New Atlantis)
- Normalizes author names and handles multiple authors
- Command-line interface with customizable creation dates and LLM model selection
- Comprehensive error handling and validation

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd article-assistant
```

2. Install dependencies with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

3. Configure LLM API keys (for generic site extraction):
```bash
# For OpenAI (recommended)
uv run llm keys set openai

# Or for Anthropic Claude
uv add llm-anthropic
uv run llm keys set anthropic

# Verify installation
uv run llm --version
```

**Note**: LLM configuration is only required for extracting metadata from generic websites. The New Atlantis extraction works without any API keys.

## Usage

### Basic Usage

**Extract metadata from a New Atlantis article** (specialized, fast, no LLM):

```bash
uv run python extract_article_metadata.py "https://www.thenewatlantis.com/publications/the-tyranny-of-now"
```

**Extract metadata from a generic website** (LLM-based):

```bash
uv run python extract_article_metadata.py "https://example.com/article-url"
```

**Output:**
```yaml
---
title: The Tyranny of Now
author:
  - Nicholas Carr
format: journal article
creation-date: 2025-06-19
publication: The New Atlantis
---

## Notes
```

### Specifying LLM Model

Choose a different LLM model for generic extraction:

```bash
# Use OpenAI GPT-4
uv run python extract_article_metadata.py --model gpt-4o "https://example.com/article"

# Use Anthropic Claude (requires llm-anthropic plugin)
uv run python extract_article_metadata.py --model claude-3.5-sonnet "https://example.com/article"
```

**Note**: The `--model` argument only affects generic site extraction. The New Atlantis uses specialized extraction logic.

### Custom Creation Date

Specify a custom creation date:

```bash
uv run python extract_article_metadata.py --creation-date 2024-12-01 "https://example.com/article-url"
```

### JSON Output

Print the metadata as a single JSON object instead of the YAML header, for scripts and pipelines:

```bash
uv run python article_assistant.py metadata --json "https://www.thenewatlantis.com/publications/zoning-alone"
```

**Output** (one line, shown wrapped here):
```json
{"title": "Zoning Alone", "authors": ["Joseph Lawler"],
 "date_published": "2026-09-08T15:39:03+00:00",
 "description": "The 100-year-old experiment of zoning is part of why Americans have become so isolated. The Supreme Court may soon bring it to an end — and help bring back real neighborhoods.",
 "issue_number": "86", "issue_season": "Fall 2026",
 "publication": "The New Atlantis"}
```

For The New Atlantis, a missing key means the value was absent from the page: a web-only piece has no `issue_number` or `issue_season`. The exception is `date_published`: when the page gives no publication date but names an edition, it is the first day of that edition's season, as a date with no time (`2025-12-01` for Winter 2026). Generic sites go through the LLM extractor, which returns only `title`, `authors`, `publication` and `date_published`; those can be empty (`""`, `[]`, `null`), and `date_published` is left out when the LLM call fails.

### Help

View all available options:

```bash
uv run python extract_article_metadata.py --help
```

## Output Format

The script generates YAML front matter compatible with common note-taking applications like Obsidian, Zettlr, and other Markdown-based tools:

```yaml
---
title: Article Title
author:
  - Author Name
  - Second Author (if applicable)
format: journal article
creation-date: YYYY-MM-DD
publication: The New Atlantis
periodical-edition: No. XX (Season YYYY)  # When available
---

## Notes
```

## Development

### Running Tests

The project includes comprehensive unit and functional tests:

```bash
# Run all tests
uv run python -m pytest

# Run with verbose output
uv run python -m pytest -v

# Run specific test file
uv run python -m pytest tests/test_extract_article_metadata.py
uv run python -m pytest tests/test_functional.py
```

### Code Quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check code style
ruff check .

# Format code
ruff format .

# Check and fix issues
ruff check . --fix
```

### Project Structure

```
article-assistant/
├── extract_article_metadata.py    # Main script
├── tests/                         # Test directory
│   ├── __init__.py               # Test package marker
│   ├── test_extract_article_metadata.py  # Unit tests
│   ├── test_extractors.py        # Extractor tests
│   └── test_functional.py        # Functional tests
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                       # Locked dependency versions
├── .gitignore                     # Git exclusion rules
├── CLAUDE.md                      # Claude Code development guidelines
└── README.md                      # This file
```

## Technical Details

### Architecture

The script uses the **Strategy pattern** with polymorphic extractors:

- **MetadataExtractor** (ABC): Abstract interface defining `extract_metadata()` and `supports_url()`
- **NewAtlantisExtractor**: Specialized extractor for The New Atlantis articles
  - JSON-LD structured data parsing (primary)
  - HTML fallback parsing (secondary)
  - Edition number inference from season/year
  - No external API calls - fast and free
- **LLMExtractor**: Generic extractor using `llm` utility
  - Structured output with Pydantic schemas
  - HTML-to-text conversion with cleanup
  - Token-efficient (4000 char limit)
  - Graceful error handling
- **get_extractor_for_url()**: Factory function for site detection routing

### Site Detection

The script automatically detects the site and uses the appropriate extractor:

1. **The New Atlantis** (`thenewatlantis.com`): Uses `NewAtlantisExtractor`
2. **All other sites**: Uses `LLMExtractor` with configured model

### Metadata Extraction

**For The New Atlantis:**
1. **Primary**: JSON-LD structured data parsing for reliable metadata extraction
2. **Fallback**: HTML parsing when structured data is unavailable or malformed
3. **Edition Inference**: Automatically calculates edition numbers from season/year information

**For Generic Sites:**
1. **HTML Cleaning**: Remove script, style, nav, header, footer tags
2. **Text Extraction**: Convert to plain text (limited to 4000 chars)
3. **LLM Extraction**: Use structured output (Pydantic schema) for validation
4. **Fallback**: Basic HTML parsing if LLM fails

### Supported Metadata Fields

- **Title**: Article headline
- **Author(s)**: Single or multiple authors with whitespace normalization
- **Publication**: Publication name or website
- **Issue Information**: Issue number and season (The New Atlantis only)
- **Date Published**: Publication date in ISO format (when available)
- **Creation Date**: Current date or user-specified date

### Edition Number Inference (The New Atlantis)

When articles only display season information (e.g., "Winter 2025") without explicit edition numbers, the script automatically calculates the correct edition number using The New Atlantis's quarterly publication schedule:

- **Publication Pattern**: Winter, Spring, Summer, Fall (4 issues per year)
- **Reference Points**: Winter 2025 = No. 79, Summer 2025 = No. 81
- **Calculation**: Supports past and future years with accurate numbering
- **Format Support**: Handles both "Season Year" and "Year Season" formats

### Error Handling

- Network request failures with informative error messages
- Graceful degradation when metadata is missing
- LLM API errors with fallback to basic HTML parsing
- Clear error messages for missing API keys
- Proper exit codes for scripting integration

## Dependencies

Dependencies are managed via [uv](https://docs.astral.sh/uv/) and defined in `pyproject.toml`.

### Core Dependencies
- **requests**: HTTP library for fetching article content
- **beautifulsoup4**: HTML parsing and DOM navigation

### Multi-Site Support Dependencies
- **llm**: Python library for LLM interactions (generic site extraction)
- **pydantic**: Data validation and structured output schemas

### Development Dependencies
- **pytest**: Testing framework
- **pytest-mock**: Mocking utilities for tests

## Troubleshooting

**Import errors**:
- Ensure dependencies are installed: `uv sync`

**Network errors**:
- Check your internet connection
- Verify the URL is accessible in a web browser
- Some sites may block automated requests

**LLM extraction errors:**

- **"llm package not installed"**:
  ```bash
  uv sync
  ```

- **"Failed to initialize LLM model" / API key errors**:
  ```bash
  # Configure API key for your chosen provider
  uv run llm keys set openai
  # Or
  uv run llm keys set anthropic
  ```

- **"Model not available"**:
  ```bash
  # List available models
  uv run llm models list

  # Install additional model plugins
  uv add llm-anthropic  # For Claude
  uv add llm-ollama     # For local models
  ```

- **Slow LLM extraction**:
  - LLM extraction is slower than specialized extraction due to API calls
  - Consider using local models via `llm-ollama` plugin for faster inference
  - The New Atlantis extraction remains fast (no LLM overhead)

- **LLM extraction accuracy**:
  - LLM extraction quality depends on the model used
  - `gpt-4o-mini` provides good balance of speed, cost, and accuracy
  - For higher accuracy, use `gpt-4o` or `claude-3.5-sonnet` with `--model` flag
  - Always verify extracted metadata manually

## Development Notes

This project was developed using [Claude Code](https://claude.ai/code), Anthropic's AI-powered development environment, which assisted with code generation, testing, and documentation.

## License

This project is provided as-is for educational and personal use.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`uv run python -m pytest`)
5. Check code quality (`ruff check . && ruff format .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Future Enhancements

- Support for additional academic journals and publications
- Configuration file for customizable output formats
- Integration with popular note-taking applications
- Batch processing of multiple articles
- Enhanced metadata extraction (DOI, citation information, etc.)

## Troubleshooting

### Common Issues

**Import errors when running the script:**
- Ensure dependencies are installed: `uv sync`

**Network timeout errors:**
- Check your internet connection
- Some articles may require authentication or have access restrictions

**Metadata extraction issues:**
- The script prioritizes JSON-LD structured data but falls back to HTML parsing
- Some older articles may have different HTML structures
- Report issues with specific URLs for investigation

### Getting Help

- Check the [Issues](https://github.com/your-username/article-assistant/issues) page for known problems
- Create a new issue with:
  - The article URL you're trying to process
  - The error message or unexpected output
  - Your Python version and operating system