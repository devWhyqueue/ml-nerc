# Word Embeddings for CRF Feature Enhancement

This directory is intended to store pre-trained word embedding files that will be used to enhance the CRF model's
feature set.

## Recommended Embeddings

For best results, download one of the following pre-trained embedding models:

1. **GloVe embeddings**: [Download from Stanford NLP](https://nlp.stanford.edu/projects/glove/)
    - Recommended file: `glove.6B.50d.txt` (50-dimensional embeddings trained on Wikipedia + Gigaword)

2. **fastText embeddings**: [Download from Facebook Research](https://fasttext.cc/docs/en/english-vectors.html)
    - Recommended file: `wiki-news-300d-1M.vec.zip` (extract the .vec file)

## Setup Instructions

1. Download your preferred embedding file
2. Place the file in this directory
3. Set the environment variable to point to your embedding file:

```bash
# For Windows PowerShell
$env:EMBEDDING_FILE="path/to/embedding/file.txt"
$env:EMBEDDING_DIM="50"  # Set to match your embedding dimensions

# For Linux/Mac
export EMBEDDING_FILE="path/to/embedding/file.txt"
export EMBEDDING_DIM="50"  # Set to match your embedding dimensions
```

If you don't set these environment variables, the system will look for `glove.6B.50d.txt` in this directory by default.

## Memory Considerations

Word embeddings can be memory-intensive. If you encounter memory issues:

1. Use a smaller embedding file (e.g., 50d instead of 300d)
2. Reduce the vocabulary size by editing the `max_vocab` parameter in `embeddings.py`
3. Use the `get_truncated_embeddings` function instead of `get_embedding_features` to use fewer dimensions

## Performance Impact

Adding word embeddings typically improves performance for:

- Rare entity classes (like drug_n)
- Entities with consistent semantic properties
- Out-of-vocabulary terms that are semantically similar to known entities
