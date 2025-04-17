# Engineered Features in the Enhanced NER System

This document provides a comprehensive explanation of the engineered features incorporated into our Conditional Random
Field (CRF)-based Named Entity Recognition (NER) system. Each feature set is designed to provide complementary insights
about the tokens (words) in a sentence to improve entity recognition accuracy.

---

## 1. Basic Token Features

These features capture the raw form and normalized versions of each token.

- **Token Form**: The original token as it appears in the text.
- **Lowercase Form**: The token converted to lowercase. This normalization helps alleviate case-sensitivity issues.
- **Prefix and Suffix Features**: For tokens of sufficient length, prefixes and suffixes of lengths 3 and 4 are extracted. These features capture sub-word patterns that may be indicative of certain entity types.

---

## 2. Character-level Features

Character-level features provide insights into the makeup of each token from the perspective of individual characters
and short sequences.

- **Character n-grams**: Extracts contiguous substrings (n-grams) of length 2. Extraction is limited to a maximum of 5 n-grams per token.
- **Word Shape**: Transforms the token into a simplified “shape”. For example, "Acetaminophen" becomes `Xxxxxxxxxxxxx`
  and "FDA" becomes `XXX`. This abstraction helps the model understand patterns like capitalization and the presence of
  digits, without relying on the exact characters.
- **Casing Pattern**: Determines the overall casing of the token (e.g., ALLCAPS, TITLE, LOWER, MIXED). This helps
  distinguish tokens based on whether they are written in upper-case, title-case, etc.

Additional checks also implemented include:

- **Has Digit**: Marks tokens that contain any digits. This can be relevant for tokens like alphanumeric identifiers.
- **Has Hyphen**: Flags tokens that contain a hyphen, common in composite drug names or multi-part entities.
- **Has Parenthesis/Bracket**: Indicates if tokens contain parentheses or brackets, which can be used to isolate
  additional details within text.
- **Chemical Pattern Detection**: Uses pre-defined regex patterns to identify tokens that match simple chemical-like structures.
- **Drug Affix Detection**: Checks for common drug-like prefixes or suffixes by normalizing the token and comparing it to known patterns.
- **Long Word Identification**: Marks tokens longer than 10 characters. Such tokens sometimes indicate complex drug
  names or other lengthy entities.

---

## 3. Contextual Features

Context is a key indicator in sequence labeling tasks. These features incorporate information from adjacent tokens to
help the model better predict the entity type:

- **Previous Token Features**: The form, lowercase form, and casing of the preceding token are incorporated. In
  addition, a bi-gram feature that combines the previous token with the current token is generated.
- **Extended Left-Context**: If available, the token two positions to the left is used to form a tri-gram feature that
  spans two previous tokens and the current token.
- **Next Token Features**: Similarly, the following token’s form, lowercase form, and casing pattern are added.
- **Extended Right-Context**: If available, the token two positions ahead is also considered.
- **Sentence Boundaries**: For tokens at the start or end of a sentence, special tokens (`BoS` for beginning of sentence
  and `EoS` for end of sentence) are introduced.

---

## 4. Lexicon-based Features

Lexicon-based features incorporate external domain knowledge from curated dictionaries of approved and not approved drug names.

- **Fuzzy Lexicon Matching**: Uses fuzzy string matching (via rapidfuzz, cutoff 85) to check if the token approximately matches any entry in the approved or not approved drug lexicons. Features `fuzzyApprovedDrugsLexicon=true` or `fuzzyNotApprovedDrugsLexicon=true` are added if a match is found.

---

## 5. Part-of-Speech (POS) Tagging Feature

Part-of-speech tagging is incorporated to provide syntactic context to each token.

- **POS Tag Feature**: Using the Natural Language Toolkit (NLTK), each token is tagged with its grammatical category (
  e.g., noun, verb, adjective). This tag is appended as a feature (e.g., `pos=NN`). If the POS tag cannot be determined
  or if an error occurs during tagging, a placeholder tag (`UNK`) is used.

---

## 6. Embedding Features

Word embedding features are extracted if pre-trained embeddings are available:

- **Discretized Embedding Bins**: For each embedding dimension, the value is binned to reduce feature space (e.g., `emb_0=2`).
- **Truncated Embeddings**: Optionally, only the first N dimensions are used for efficiency (e.g., `emb_0=0.1234`).
- **OOV Handling**: If a token is out-of-vocabulary, `emb_OOV=true` is used. If embeddings are unavailable,
  `emb_unavailable=true` is set.

Embeddings provide dense, semantic information about tokens, augmenting the surface-level and lexicon-based features.

---

## Summary

The feature engineering approach in this system is designed to offer a rich, multi-faceted representation of each token.
By combining:

- **Surface-level features** (token forms, prefixes, and suffixes),
- **Character-level patterns** (n-grams, word shape, casing),
- **Contextual cues** (neighboring tokens and sentence boundaries),
- **Domain-specific lexicon lookups**, and
- **Syntactic information** (POS tags),

the system aims to provide robust inputs for the CRF model, enhancing its ability to identify and classify named
entities effectively. This multi-layered feature setup is particularly valuable for complex tasks such as recognizing
pharmacological entities where both morphology and context play key roles.
