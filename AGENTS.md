# Repository Guidelines

## Project Structure and Module Organization
This repository is a small collection of Python scripts and reference text. The root directory contains the runnable examples:
- `gemini_chat.py`: example request to a Gemini-style endpoint.
- `get_token.py`, `get_tokens.py`, `new_token.py`, `search_token.py`: token management API examples.
- `guideline.txt`, `user_info.txt`: reference notes and links.

There are no nested packages or test directories; scripts are designed to run directly from the repo root.

## Build, Test, and Development Commands
There is no build system. Run scripts directly with Python:
- `python gemini_chat.py`: sends an example multimodal request.
- `python get_tokens.py`: lists tokens via the API.
- `python new_token.py`: creates a new token using the payload in the script.

All scripts use placeholders (e.g., `<token>`, `<api-key>`, `HTTPSConnection("")`) that must be updated before running.

## Coding Style and Naming Conventions
Follow standard Python style (PEP 8), 4-space indentation, and snake_case for variables. Keep scripts short and focused on a single API workflow. Prefer explicit variable names like `payload`, `headers`, and `conn` to match existing files.

## Testing Guidelines
No automated tests are present. If you add tests, place them in a new `tests/` directory and use `pytest` with file names like `test_tokens.py`. Keep tests isolated from live API calls by mocking HTTP responses.

## Commit and Pull Request Guidelines
There is no git history in this folder, so no commit message conventions are defined. Use clear, imperative summaries (e.g., "Add token search example") and include:
- A short description of the change and any API endpoints touched.
- Sample usage or updated placeholders if behavior changes.

## Security and Configuration Tips
Do not commit real API keys or tokens. Keep credentials in environment variables or local-only edits. When sharing examples, keep `<token>` and `<api-key>` placeholders intact.
