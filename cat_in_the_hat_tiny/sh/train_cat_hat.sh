#!/bin/zsh
#
# train_cat_hat.sh - Tiny transformer on The Cat In The Hat
#
# n_embd=4: visualizable as 3D + color, or two 2D plots.
# n_head=2: each head gets 2 dimensions.
# n_layer=16: deep to compensate for narrow embeddings.
#
# Previous runs:
#   n_embd=3, n_layer=2  (1,260 params): plateaued at loss 4.0
#   n_embd=3, n_layer=16 (3,318 params): plateaued at loss 3.8
#   Both too few params. n_embd=3 is the bottleneck, not depth.
#
# NOTE: Run this on the M3, not the Mac Studio.
#

sh/train.sh \
    --input txt_local/the_cat_in_the_hat_2b_lower.txt \
    --output pt/cat_hat_4d.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 256 \
    --n_layer 16 \
    --n_head 2 \
    --n_embd 4 \
    --block_size 64 \
    --batch_size 4 \
    --max_iters 50000 \
    --learning_rate .001 \
    --warmup_iters 100 \
    --val_split 0.1 \
    --dropout 0.0
