# llm-synth-training-data
A small repo to store my more recent LLM training scripts and data for synthing.

`./scripts/`: files for creating synth data

- `langchain_gen_synth_data.py` - A small langchain script for generating chat data, can be used with ollama or any other provider

`./data/`: output files from the scripts

- `other_conversations.jsonl`: Conversations to give my llm a persona as created by `langchain_gen_synth_data.py`
