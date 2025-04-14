Below is a markdown file that explains in detail each of the engineered features present in the enhanced NER feature
extraction system.

---

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

---

This explanation provides an overview of how each engineered feature contributes to the overall NER system's performance
and effectiveness.
