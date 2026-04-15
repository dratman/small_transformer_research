# Small Transformer Research

## What this project is

Research into the internals of small GPT-type language models. The goal is
to understand how these models work internally so humans can learn to
interpret and improve them. Built on 80+ diary entries of mechanistic
interpretability research on character-level transformers.

Currently training and comparing transformer models with identical
architecture but different tokenization (character-level vs BPE) and
different FFN types (standard vs gated SwiGLU).

## The three models

### Model B — Character-level (reference, training complete)
- Architecture: 16 layers, 16 heads, 2048 embedding dim, softmax attention
- Tokenizer: character-level, vocab size 52
- Parameters: ~822M
- Trained to iter 53,500, best val loss: 0.8322
- Corpus: gutenberg_corpus_MODERN_CLEAN.txt
- Checkpoint: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous.pt`

### BPE model (training on Mac Studio as of 2026-04-13)
- Architecture: 16 layers, 16 heads, 2048 embedding dim, softmax attention
- Tokenizer: BPE, vocab size 8,192
- Parameters: ~822M
- Corpus: corpus_of_gutenberg_novels_cleaned_shuffled_2026_03_03_B.txt
  (8.3 billion characters, 2.03 billion BPE tokens)
- Note: corpus quality issues discovered (non-fiction, encoding artifacts).
  A cleaned corpus is being prepared. See diary 082_B and corpus rebuild
  section below.
- Checkpoint: `pt/bpe_16L16H.pt`
- Training log: `terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt`
- Training speed: ~0.20 iters/sec (~700 iters/hour)

### Gated-FFN model (planned, not yet training)
- Architecture: 16 layers, 16 heads, 2048 embedding dim, softmax attention
- FFN: **Gated SwiGLU** (use_gated_mlp=True), intermediate_size=5376
- Tokenizer: BPE, vocab size 8,192
- Parameters: ~814M
- Purpose: LARQL-compatible model for queryable internals
- Will train on the cleaned corpus once ready
- See diary 082_B for design rationale

## Corpus rebuild (in progress as of 2026-04-15)

The current training corpus contains non-fiction reference works, religious
texts, and encoding artifacts. A cleaned version is being built:

1. `py/filter_corpus.py` — classifies 8,794 Gutenberg texts as keep/remove.
   Output: `corpus_keep.txt` (7,543 files), `corpus_remove.txt` (1,251 files)
2. `py/scan_corpus_quality.py` — scans kept files for dialect and noise.
   Output: `corpus_quality_report.txt` (192 files flagged)
3. `py/rebuild_corpus.py` — builds final corpus from kept files:
   strips headers, lowercases, cleans noise, shuffles paragraphs.
   Final count: 7,351 files after quality exclusions.

To run the rebuild (on Mac Studio where source texts are local):
```
python py/rebuild_corpus.py \
    --texts_dir /path/to/Gutenberg_Project_Books/gutenberg_texts \
    --keep corpus_keep.txt \
    --quality_report corpus_quality_report.txt \
    --output txt_local/corpus_cleaned_2026_04_15.txt
```

Source texts: 8,794 individual Gutenberg files in
`Gutenberg_Project_Books/gutenberg_texts/` (on iCloud, accessible from
Mac Studio and M3 MacBook).

## Comparing losses between models

BPE per-token loss CANNOT be compared directly to character per-token loss.
The BPE model predicts 1 of 8,192 tokens; the character model predicts 1
of 52 characters. To compare them, normalize to per-character loss:

    BPE per-character loss = BPE per-token loss / (chars per token)

The corpus has 8.3B chars / 2.03B tokens = ~4.09 characters per BPE token.

Example: BPE loss 3.36 per token ≈ 0.82 per character, comparable to
Model B's 0.8322 per character.

## Key diary entries for context

The `diary/` directory contains 83+ research entries. Key entries:

- **082_B**: LARQL and gated-FFN model plan
- **083_B**: Repository creation and file consolidation
- **081**: BPE vs char comparison setup and first results
- **080**: Thermodynamics of training — layer crystallization order
- **076**: Layer-by-layer activation analysis of a character model
- **074**: Persistent homology — topological crystallization
- **055**: Dark subspace — exists in char models (vocab << dim), absent
  in BPE models (vocab > dim)
- **049**: Synthesis of character model findings

## Project structure

- `py/` — Python source:
  - `model.py` — GPT model with standard and gated SwiGLU MLP options
  - `train.py` — training script
  - `tokenizer.py` — BPE tokenizer
  - `sample.py` — text generation
  - `filter_corpus.py` — classify Gutenberg texts as keep/remove
  - `scan_corpus_quality.py` — scan for dialect and character noise
  - `rebuild_corpus.py` — build cleaned corpus from filtered texts
  - `clean_session_log.py` — strip ANSI codes from terminal session logs
- `sh/` — Shell scripts: train.sh (wrapper), train_bpe_16L16H.sh
- `diary/` — Research diary entries 001-083
- `doc/` — Chat transcripts, specs
- `sft_data/` — SFT Q&A pairs
- `corpus_keep.txt` — 7,543 Gutenberg files to include in corpus
- `corpus_remove.txt` — 1,251 files to exclude with reasons
- `corpus_quality_report.txt` — dialect and noise scan results
- `pt/` — Checkpoints (gitignored, large files)
- `txt_local/` — Training corpus (gitignored, ~8 GB)
- `terminal_logs/` — Training output logs (gitignored)

## Multi-machine setup

This project spans three Macs. The GitHub repo
(https://github.com/dratman/small_transformer_research) is the single
source of truth for code, diary entries, and scripts. Large files
(checkpoints, corpus) stay local.

- **M2 MacBook** — this machine, analysis and development
- **Mac Studio M2 Ultra** — primary training machine (192GB RAM)
- **M3 MacBook Pro** — has Gutenberg source texts locally

## Technical notes

- The BPE tokenizer crashes on strings >~1GB. The encode() method in
  tokenizer.py chunks large texts into 100 MB pieces, splitting on
  newline boundaries. This was fixed on 2026-04-05.
- train.sh uses the --output filename (not --input) for log naming.
- Training uses python -u for unbuffered output to logs.
- model.py supports use_gated_mlp=True for SwiGLU FFN (LARQL-compatible).
  Standard MLP is the default for backward compatibility.
- LARQL (https://github.com/chrishayuk/larql) requires gated FFN models.
  See diary 082_B for compatibility analysis.
