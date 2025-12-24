# Repository Guidelines

## Project Structure & Module Organization
The site is a simple static build with everything at the repository root.
- `.github/` stores GitHub metadata and workflows.
- `index.html` is the home page.
- `data-structures.html`, `algorithms.html`, and `patterns.html` are the category pages.
- `styles.css` contains all styling.
- `script.js` stores the content data and rendering logic.

Keep new assets at the top level unless there is a strong reason to add a directory, and document any new directories here and in `README.md`.

## Build, Test, and Development Commands
No build or test commands are defined. The site is static and can be opened directly in the browser.

## Coding Style & Naming Conventions
- Markdown: sentence-case headings, one blank line between sections, and short paragraphs.
- Python snippets: PEP 8, 4-space indentation, `snake_case` names.
- Filenames: lowercase `kebab-case` with topic/context. Avoid spaces.

## Testing Guidelines
There are no tests yet. If you add tests, place them under `tests/` and use clear names such as `test_sorting.py` or `render.spec.ts` depending on language. Focus coverage on algorithm correctness and any rendering/visualization helpers.

## Commit & Pull Request Guidelines
Recent history uses short, descriptive messages (e.g., "Initial plan"). Keep subjects concise, capitalized, and focused on one change. For pull requests, include a summary, link related issues, and add screenshots or GIFs for visual changes.

## Security & Configuration Tips
Do not commit secrets. If configuration becomes necessary, add an `.env.example` with placeholders and keep real values out of the repository.
