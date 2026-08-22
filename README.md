# Recipe Chain

Small educational Python project: generate a recipe from a natural-language request using a **four-stage LLM prompt chain** built with LangChain and OpenAI.

The scope is intentionally narrow — it demonstrates prompt decomposition and sequential chaining, not a production cooking assistant.

## What it demonstrates

A complex task (“create a recipe that fits my constraints”) is split into four focused LLM calls instead of one large prompt:

| Stage | Function | Input | Output |
|-------|----------|-------|--------|
| 1 | `build_analysis_chain` | User request | Structured analysis (ingredients, constraints, expectations) |
| 2 | `build_strategy_chain` | Analysis | Cooking strategy (technique, steps, equipment) |
| 3 | `build_recipe_chain` | Analysis + strategy | Recipe draft with sections |
| 4 | `build_review_chain` | **Original request + draft** | Final recipe with a quality-review section |

`run_chain(user_request)` runs the stages in order and passes each output to the next step. The review stage receives both the original request and the draft so it can check constraint compliance.

## Project structure

```
langchain-recipe-chain-homework/
├── recipe_chain.py              # Main script and four chains
├── requirements.txt             # Pinned major-version bounds
├── .env.example                 # Environment variable template
├── .gitignore
├── README.md
└── examples/
    ├── sample_chain_steps.md    # Committed example (full step-by-step trace)
    ├── chain_steps.md           # Generated at runtime (gitignored)
    └── sample_output.md         # Final recipe only, generated at runtime (gitignored)
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment configuration

Copy the template and set your API key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

`.env` contents:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.2
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (**required** for a real run) |
| `OPENAI_MODEL` | Model name; default `gpt-4o-mini` |
| `TEMPERATURE` | Sampling temperature; default `0.2` |

## How to run

```bash
python recipe_chain.py "Хочу быстрый ужин из курицы, риса и овощей, без духовки, до 30 минут"
```

The script prints stage logs `[1/4]` … `[4/4]` and the final recipe in the terminal.

Without arguments it prints usage help.

## Sample output vs runtime artifacts

| File | Role |
|------|------|
| `examples/sample_chain_steps.md` | **Committed** example showing all four stages — safe to browse in the repo |
| `examples/chain_steps.md` | Written on each run; **gitignored** |
| `examples/sample_output.md` | Final recipe only from the last run; **gitignored** |

Do not commit runtime-generated files.

## Limitations

- **Sequential LLM calls** — four separate API requests per run (latency and cost scale linearly).
- **No deterministic validation** — there is no rule engine or schema check; quality depends on the model.
- **Model-based review is not a guarantee** — the final review step can miss or misjudge constraint violations.
- **Requires an OpenAI API key** — the script exits with a clear error if `OPENAI_API_KEY` is missing.
- **Educational scope** — not suitable as-is for production use (no tests, no UI, no persistence).

## Educational note

This repository is a learning example for LangChain LCEL prompt chaining (`ChatPromptTemplate | llm | StrOutputParser()`). The same decomposition pattern applies to other domains (travel planning, technical docs, etc.).

There is no open-source license file in this repository; treat it as a personal educational portfolio piece unless a license is added later.
