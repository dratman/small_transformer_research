# Cat In The Hat — 3D Embedding Experiment

## FIRST: Read HANDOFF.md

At the start of every session, read `HANDOFF.md` for current state and
pending tasks. During your session, update it at milestones. Before
your session ends, make sure it is current and committed.

## What this project is

A transformer with n_embd=3, trained on The Cat In The Hat (~7 KB of
text). The point is that every token embedding is a 3D coordinate,
so we can directly visualize what the model learns about word
relationships. This is a microscope version of Ralph's larger project
training 822M-parameter models on Gutenberg literature.

## About Ralph

Ralph is a 74-year-old retired software/hardware engineer who was most
active in the 1980s. He works on M2 and M3 Macs.

Ralph is working on the internals of very small GPT-type language
models. His goal is to understand how humans can learn to understand
these small models and bring them to more useful functions.

## User preferences

Ralph prefers explicit over implicit communication. Assume the reader
has no prior context. State things directly.

Start every reply with the current local date and time (year, month,
day, hour, minute — no seconds).

## Problem-solving rules

When you believe you cannot do something, check whether your available
tools offer a workaround before telling the user it is impossible.

## Key questions to investigate

1. Do related words cluster in 3D? (cat/fish/things vs the/and/of)
2. Does the model learn rhyme? (hat/cat/mat/sat/that)
3. What do the attention patterns look like?
4. Can it generate Cat-In-The-Hat-style text?

## Project structure

- `py/` — model.py, train.py, tokenizer.py, sample.py
- `sh/` — train.sh (wrapper), train_cat_hat.sh (launch script)
- `diary/` — research diary
- `txt_local/` — the text
- `pt/` — checkpoints (gitignored)
- `terminal_logs/` — training logs
- `plots/` — generated plots (gitignored)

## Related projects

The larger BPE model comparison project is at:
`~/0_Home_Folder_Working_Mac_Studio/bpe_vs_char_model_comparison/`
(Mac Studio) — see diary entries 081-085 there.
