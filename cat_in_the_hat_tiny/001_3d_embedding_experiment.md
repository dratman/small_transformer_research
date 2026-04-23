# Diary 001: 3D Embedding Experiment — The Cat In The Hat

Date: 2026-04-23

## Motivation

Ralph's research program is about understanding the internals of small
GPT-type language models. The question: what do the internal
representations actually look like?

Most transformer models use embedding dimensions of 768, 2048, or
higher — far too many dimensions to visualize directly. This experiment
uses **n_embd=3**, which means every token embedding is a point in 3D
space. After training, we can plot every word in the vocabulary as a
3D scatter plot and directly see what the model has learned about word
relationships.

## Why The Cat In The Hat

- Very small vocabulary (~427 unique words, 1,561 total words)
- Highly repetitive — "said the cat", "the cat in the hat", "thing
  one and thing two" repeat many times, giving the model clear patterns
- Simple grammar — Dr. Seuss uses short sentences and heavy rhyme
- The entire text is 7 KB — training takes minutes, not days
- The repetition means even a tiny model should learn something

## Architecture

- **n_embd: 3** — the key choice. Every embedding is (x, y, z).
- **n_layer: 2** — minimal depth
- **n_head: 1** — must divide n_embd (3/1=3 dims per head)
- **vocab_size: 256** — BPE tokens, enough to cover all words
- **block_size: 64** — about 40 words of context
- **batch_size: 4**
- **learning_rate: 0.001** — higher than usual, small model can handle it
- **max_iters: 50,000**
- **Total parameters: ~10K-20K** — small enough to print every weight

## What to look for

1. **Do semantically related words cluster in 3D space?** E.g., do
   "cat", "fish", "things" form distinct regions? Do function words
   ("the", "and", "of") cluster separately from content words?

2. **Does the model learn rhyme structure?** Words that rhyme ("hat",
   "cat", "mat", "that", "sat") share sound patterns. Do they end up
   near each other in embedding space?

3. **What do the attention weights look like?** With only 1 head and
   2 layers, we can visualize the full attention pattern for any input.

4. **Can the model generate recognizable Cat-In-The-Hat-style text?**
   With such a tiny model and tiny corpus, even partial success would
   be interesting.

## Connection to larger research

This experiment is a microscope version of the BPE model comparison
project running on the Mac Studio (diary entries 081-085 in the
bpe_vs_char_model_comparison repo). That project trains 822M-parameter
models with n_embd=2048. The patterns found here in 3D may correspond
to structures that exist in the high-dimensional models but can't be
directly visualized.

Ralph's stop-words-only training idea (future experiment) connects
here too: if function words cluster distinctly from content words in
3D, that's evidence that the model separates syntax from semantics
even at this tiny scale.

## Files

- `txt_local/the_cat_in_the_hat_2b_lower.txt` — lowercased text
- `sh/train_cat_hat.sh` — launch script
- `py/` — model, training, sampling, tokenizer code (shared with
  bpe_vs_char_model_comparison project)
