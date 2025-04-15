# Engineered Features in the Enhanced NER System

This document provides a comprehensive explanation of the engineered features incorporated into our Conditional Random
Field (CRF)-based Named Entity Recognition (NER) system. Each feature set is designed to provide complementary insights
about the tokens (words) in a sentence to improve entity recognition accuracy.

---

## 1. Basic Token Features

These features capture the raw form and normalized versions of each token.

- **Token Form**: The original token as it appears in the text.
- **Lowercase Form**: The token converted to lowercase. This normalization helps alleviate case-sensitivity issues.
- **Prefix and Suffix Features**: For tokens of sufficient length, prefixes and suffixes of lengths 3, 4, and 5 are
  extracted. These features capture sub-word patterns that may be indicative of certain entity types (e.g., drug names
  often have common prefixes/suffixes).

---

## 2. Character-level Features

Character-level features provide insights into the makeup of each token from the perspective of individual characters
and short sequences.

- **Character n-grams**: Extracts contiguous substrings (n-grams) of length 2. The extraction is limited to a maximum of
  5 n-grams per token to maintain performance. These n-grams are useful to capture local character patterns.
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
- **Chemical Pattern Detection**: Uses pre-defined regex patterns to identify tokens that match known chemical
  structures.
- **Drug Affix Detection**: Checks for common drug-like prefixes or suffixes by normalizing the token and comparing it
  to known patterns.
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

Lexicon-based features incorporate external domain knowledge from curated dictionaries such as DrugBank and HSDB.

- **Normalized Token Matching**: Tokens are normalized (by lowercasing and removing simple plural endings) before
  matching against the lexicons.
- **Exact Match**: Checks if the normalized token exists in either the DrugBank or HSDB lexicon. When an exact match is
  found, a feature is added (e.g., `exactInDrugBank=true`).
- **Drug Type Feature**: If the normalized token is found in the DrugBank lexicon along with a corresponding type, that
  type is recorded.
- **Window-based Matching**: Considers two- and three-token windows to capture multi-word entities. If these token
  groups match entries in the lexicons, relevant features are set.
- **Partial Morphological Matching**: Applies a simple morphological normalization (removing common plural endings and
  other suffixes) and then checks if there is any significant substring match between the token and a list of short drug
  names. This feature is particularly useful when the surface forms vary slightly but represent the same underlying
  drug.

---

## 5. Part-of-Speech (POS) Tagging Feature

Part-of-speech tagging is incorporated to provide syntactic context to each token.

- **POS Tag Feature**: Using the Natural Language Toolkit (NLTK), each token is tagged with its grammatical category (
  e.g., noun, verb, adjective). This tag is appended as a feature (e.g., `pos=NN`). If the POS tag cannot be determined
  or if an error occurs during tagging, a placeholder tag (`UNK`) is used.

---

## 6. Drug N (Non-Proprietary Drug Name) Features

These features are specialized for recognizing non-proprietary (generic) drug names, leveraging an expanded lexicon and
FDA-derived patterns:

- **Lexicon Membership**:
  - `exactDrugNMatch=true`: Exact normalized token match in the curated drug N lexicon.
  - `drugNLexiconMatch=true`: Token found in drug N lexicon.
  - `partialDrugNMatch=true`: Partial match based on substring or pattern checks.
- **Prefix/Suffix Indexing**:
  - `drugNSuffix_*`: Token has a common drug N suffix (e.g., `drugNSuffix_ine=true`).
  - `drugNHasCommonSuffix=true`: Token has a known suffix.
  - `drugNPrefix_*`: Token has a common drug N prefix (e.g., `drugNPrefix_ace=true`).
  - `drugNHasCommonPrefix=true`: Token has a known prefix.
- **FDA Pattern Matching**:
  - `drugNPatternMatch=true`: Token matches any FDA-derived regex pattern.
  - `drugNPattern{i}=true`: Token matches the i-th FDA-derived pattern.
- **Partial Morphological Matching**: Looks for significant substring matches after normalization, to catch variations
  of drug names.
- **Miscellaneous Drug N Features**:
  - `drugNTypicalLength=true`: Normalized token length between 3 and 15.
  - `drugNMixedWithNumbers=true`: Token contains both letters and digits.
  - `drugNMultiWord=true`: Token contains multiple word parts (e.g., "alpha lipoic acid").
  - `drugNCapitalized=true`: Token is capitalized (first letter uppercase).
  - `drugNHasHyphen=true`: Token contains a hyphen.
  - `drugNHasAlphanumericHyphen=true`: Token contains a hyphen between letters and digits (e.g., "SCH-23390").
  - `drugNHasRomanNumeral=true`: Token contains a space followed by a Roman numeral (e.g., "buforin II").
  - `drugNHasChemicalPattern=true`: Token matches chemical-like patterns (e.g., "1,2-dimethyl").
  - `drugNIsAbbreviation=true`: Token is all uppercase and 2-5 characters long.
  - `drugNKnownEntity=true`: Token contains a substring of a known problematic drug entity (e.g., "mptp").

These features are designed for high recall and precision, and are optimized for performance on large lexicons.

---

## 7. Embedding Features

Word embedding features are extracted if pre-trained embeddings are available:

- **Discretized Embedding Bins**: For each embedding dimension, the value is binned to reduce feature space (e.g.,
  `emb_0=2`).
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
