# Handoff Document

Last updated: 2026-04-23 10:15 by Claude Opus (Mac Studio session)

## Current State

- **No training has been run yet.**
- **The text file is NOT in the repo** (copyright). It must be
  provided manually at: `txt_local/the_cat_in_the_hat_2b_lower.txt`
  Ralph has a copy on the Mac Studio at:
  `~/0_Home_Folder_Working_Mac_Studio/tiny_transformer/txt_local/the_cat_in_the_hat_2b_lower.txt`
- Once the text is in place, launch training with: `sh/train_cat_hat.sh`
- Training should take minutes on the M3 MacBook.

## TODO

- **TODO: Copy the text file** into `txt_local/` before training.
  Verify it exists: `ls txt_local/the_cat_in_the_hat_2b_lower.txt`
- **Run training** and verify loss decreases
- **Extract and plot 3D embeddings** after training — see README.md
  for how to extract embeddings from the checkpoint
- **Sample from the model** and evaluate output quality
- **Investigate the key questions** listed in CLAUDE.md

## Architecture choices

- n_embd=3 was chosen so embeddings can be visualized in 3D
- n_head=1 because it must divide n_embd=3
- n_layer=2 for minimal depth
- vocab_size=256 BPE tokens — covers all ~427 unique words plus subwords
- block_size=64 — about 40 words of context
- learning_rate=0.001 — higher than typical, small model tolerates it
- max_iters=50,000 — may be too many or too few, adjust as needed

## Origin

This experiment was set up by a Claude Opus instance on the Mac Studio
on 2026-04-23. The code (model.py, train.py, etc.) was copied from the
bpe_vs_char_model_comparison project. Ralph wants to run this on the
M3 MacBook (64 GB RAM).
