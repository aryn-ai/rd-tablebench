# RD Table Bench Invocation + Grading Code

This repo contains the code for invoking each provider and grading the results. This is a fork of Reducto's original repo https://github.com/reductoai/rd-tablebench with improved support for different LLM providers (OpenAI, Anthropic, Gemini).

The proprietary models that Reduco implemeted have not been touched and will not working with the grading cli.

## Installing Dependencies

```
uv pip install -r requirements.txt
```

## Downloading Data

```
uv run download_data.py
```
## Env Vars

Create an `.env` file with the following:

```
OUTPUT_DIR=data/rd-tablebench/providers
INPUT_DIR=data/rd-tablebench/pdfs

# note: only need keys for providers you want to use
ARYN_TEST_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
...
```

## Parsing

`python -m providers.llm --model gemini-2.0-flash-exp --num-workers 10`

Modify the PROVIDER variable at the top of each file before running. This is a name for a model/configuration. This name is used in each file as well as the final output directory in data/rd-tablebench/providers.

Running DocParse pipeline:
`uv run docparse.py -> uv run task_result.py -> uv run partition_to_html.py`

Each step will continue from intermediate results. You can safely rerun any step.

## Grading

`uv run python -m grade_cli --model <PROVIDER>`
