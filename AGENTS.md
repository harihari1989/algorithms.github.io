# Repository Guidelines

## Project Structure & Module Organization
The repository is currently minimal, with a root-level `README.md`. When adding site sources, keep a clear top-level layout and document each directory in `README.md` and this file. Suggested conventions:
- `src/` for algorithm writeups, demos, or site content.
- `assets/` for diagrams, images, and other media.
- `tests/` for automated checks.
- `scripts/` for build or maintenance helpers.
Group algorithms by topic (e.g., `src/sorting/`, `src/graphs/`) to keep navigation predictable.

## Build, Test, and Development Commands
No build or test commands are defined yet. If you introduce a build system, expose a single entry point (prefer `Makefile` or `package.json` scripts) and list the commands here.
Example (if introduced):
- `make dev` starts a local preview server.
- `make build` produces static output (e.g., `dist/`).
- `make test` runs tests and linters.

## Coding Style & Naming Conventions
- Markdown: sentence-case headings, one blank line between sections, and short paragraphs.
- Python snippets: PEP 8, 4-space indentation, `snake_case` names.
- Filenames: lowercase `kebab-case` with topic/context (e.g., `binary-search-step-1.svg`). Avoid spaces.

## Testing Guidelines
There are no tests yet. If you add tests, place them under `tests/` and use clear names such as `test_sorting.py` or `render.spec.ts` depending on language. Focus coverage on algorithm correctness and any rendering/visualization helpers.

## Commit & Pull Request Guidelines
Recent history uses short, descriptive messages (e.g., "Initial plan"). Keep subjects concise, capitalized, and focused on one change. For pull requests, include a summary, link related issues, and add screenshots or GIFs for visual changes.

## Security & Configuration Tips
Do not commit secrets. If configuration becomes necessary, add an `.env.example` with placeholders and keep real values out of the repository.
