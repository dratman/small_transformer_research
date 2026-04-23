#!/bin/zsh
#
# train_cat_hat.sh - Tiny transformer on The Cat In The Hat
#
# n_embd=3: every word embedding is a 3D point you can visualize.
# n_layer=16: deep network to compensate for narrow embeddings.
# Previous run with n_layer=2 had only 1,260 parameters and
# plateaued at loss 4.0 — too few parameters to learn the text.
#
# vocab_size 256: BPE with enough tokens to cover all words + subwords
# n_head 1: must divide n_embd=3
# block_size 64: ~40 words of context
# batch_size 4: small text, small batches
# learning_rate 0.001: higher lr for tiny model
#
# NOTE: Run this on the M3, not the Mac Studio (which is running
# the large BPE training).
#

sh/train.sh \
    --input txt_local/the_cat_in_the_hat_2b_lower.txt \
    --output pt/cat_hat_3d.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 256 \
    --n_layer 16 \
    --n_head 1 \
    --n_embd 3 \
    --block_size 64 \
    --batch_size 4 \
    --max_iters 50000 \
    --learning_rate .001 \
    --warmup_iters 100 \
    --val_split 0.1 \
    --dropout 0.0
