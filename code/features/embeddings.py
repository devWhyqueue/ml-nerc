#!/usr/bin/python3
"""
Word embedding features for CRF-based NER.
Loads pre-trained word embeddings and provides functions to extract features from them.
"""
import os
import sys

import numpy as np


def load_embeddings(filepath, dimension=50, max_vocab=50000):
    """
    Load pre-trained word embeddings from a file.
    
    Args:
        filepath: Path to the embedding file (e.g., glove.6B.50d.txt)
        dimension: Dimension of the embeddings
        max_vocab: Maximum vocabulary size to load (to limit memory usage)
        
    Returns:
        Dictionary mapping words to their embedding vectors
    """
    embeddings = {}

    if not os.path.exists(filepath):
        print(f"Warning: Embedding file {filepath} not found. Skipping embeddings.", file=sys.stderr)
        return embeddings

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_vocab:
                    break

                parts = line.strip().split()
                if len(parts) <= dimension:
                    continue

                word = parts[0]
                vec = np.array(parts[1:dimension + 1], dtype=float)
                embeddings[word] = vec

        print(f"Loaded {len(embeddings)} word embeddings of dimension {dimension}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading embeddings: {e}", file=sys.stderr)

    return embeddings


def get_embedding_features(token, embeddings, dimension=50, bins=5):
    """
    Extract features from word embeddings for a token.
    
    Args:
        token: The token to get features for
        embeddings: Dictionary mapping words to embedding vectors
        dimension: Dimension of the embeddings
        bins: Number of bins to discretize embedding values
        
    Returns:
        List of embedding features
    """
    t_lower = token.lower()

    # If embeddings dictionary is empty, return OOV feature
    if not embeddings:
        return ["emb_unavailable=true"]

    if t_lower in embeddings:
        emb = embeddings[t_lower]

        # Option 1: Use discretized values (reduces feature space)
        features = []
        for i in range(min(dimension, len(emb))):
            # Discretize the embedding value into bins
            bin_value = int(np.floor((emb[i] + 1.0) * (bins / 2.0)))
            bin_value = max(0, min(bins - 1, bin_value))  # Clamp to valid bin range
            features.append(f"emb_{i}={bin_value}")

        return features
    else:
        # Out-of-vocabulary handling
        return ["emb_OOV=true"]


def get_truncated_embeddings(token, embeddings, num_dims=10):
    """
    Get a truncated version of the embedding (first N dimensions only).
    Useful when full embeddings are too large for the CRF.
    
    Args:
        token: The token to get features for
        embeddings: Dictionary mapping words to embedding vectors
        num_dims: Number of dimensions to include
        
    Returns:
        List of embedding features with only the first num_dims dimensions
    """
    t_lower = token.lower()

    if not embeddings:
        return ["emb_unavailable=true"]

    if t_lower in embeddings:
        emb = embeddings[t_lower]
        return [f"emb_{i}={round(emb[i], 4)}" for i in range(min(num_dims, len(emb)))]
    else:
        return ["emb_OOV=true"]
