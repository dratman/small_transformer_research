# Cat In The Hat — Tiny 3D Transformer

A transformer with n_embd=3, trained on The Cat In The Hat.
Every word embedding is a point in 3D space that you can visualize.

## Setup

Transfer this entire directory to the M3 MacBook.

## Run training

    cd cat_in_the_hat_tiny
    sh/train_cat_hat.sh

## Sample from the model

    python py/sample.py --model pt/cat_hat_3d.pt --prompt 'the cat'

## What to look at

After training, the embeddings are in the checkpoint. To extract
and plot them in 3D:

    python -c "
    import torch
    checkpoint = torch.load('pt/cat_hat_3d.pt', map_location='cpu', weights_only=False)
    embeddings = checkpoint['model']['transformer.wte.weight']
    print(f'Shape: {embeddings.shape}')  # [256, 3]
    # Each row is a 3D point — one per BPE token
    "

## Parameters

- Model: ~10K-20K parameters total
- Vocab: 256 BPE tokens
- Embedding: 3 dimensions
- Layers: 2
- Heads: 1
- Context: 64 tokens
- Text: 1,561 words, 427 unique, 7,067 characters
