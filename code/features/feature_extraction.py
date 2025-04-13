#!/usr/bin/python3
"""
Enhanced feature extraction for a CRF-based NER system.

This script adds:
1. POS tagging features via NLTK.
2. Word shape features (e.g., "Xx" for capital-lower).
3. Expanded partial match logic that considers morphological variants.
4. Improved lexicon usage (e.g., normalizing tokens before lookup).
"""

import sys

import nltk


# Make sure you have downloaded the NLTK 'averaged_perceptron_tagger' model:
#   import nltk
#   nltk.download('averaged_perceptron_tagger')

# You can import or copy over the existing code (casing, char_ngrams, etc.) here from your original version.
# Below, we keep or reuse many original methods and add new improvements.

##################################################
# Original helper functions (with some enhancements)
##################################################

def casing(token):
    """Determine the casing pattern of a token."""
    if token.isupper():
        return "ALLCAPS"
    elif token.istitle():
        return "TITLE"
    elif token.islower():
        return "LOWER"
    elif any(c.isupper() for c in token):
        return "MIXED"
    else:
        return "NOCASE"


def char_ngrams(token, n):
    """Extract limited character n-grams."""
    if len(token) < n:
        return [token.lower()]
    # Keep up to 5 n-grams for performance
    ngrams = []
    for i in range(min(5, len(token) - n + 1)):
        ngrams.append(token[i: i + n].lower())
    return ngrams


def has_chemical_pattern(token, chemical_patterns):
    """Check if token matches any known chemical pattern regex."""
    return any(pattern.search(token) for pattern in chemical_patterns)


def has_drug_affix(token, common_drug_prefixes, common_drug_suffixes):
    """Check if token has drug-like prefix or suffix."""
    t_lower = token.lower()
    if len(t_lower) < 3:
        return False
    return (t_lower[:3] in common_drug_prefixes or t_lower[-3:] in common_drug_suffixes)


def word_shape(token):
    """
    Return a simplified 'shape' of the token to capture 
    letter/digit/uppercase/length patterns, e.g.:
       - "Acetaminophen" -> "Xxxxxxxxxxxxx"
       - "FDA"           -> "XXX"
       - "CYP2D6"        -> "XXXdX"
    """
    shape_str = []
    for char in token:
        if char.isdigit():
            shape_str.append("d")
        elif char.isalpha():
            if char.isupper():
                shape_str.append("X")
            else:
                shape_str.append("x")
        else:
            shape_str.append(char)
    # Optionally collapse repeating shape symbols (less granular):
    # e.g. "Xxxxx" -> "Xx+"
    # For now, we’ll keep the full shape.
    return "".join(shape_str)


##################################################
# New partial matching improvements
##################################################

def partial_match_morphological(token, short_drugs, cache):
    """
    Extended partial match check that tries simple morphological variants.
    E.g., removing trailing 's', 'es', etc., or a basic lemma-like approach.
    """
    base_candidate = token.lower().rstrip(".,;:!?-").replace("'s", "")

    # Remove common plural suffixes
    if base_candidate.endswith("s"):
        base_candidate = base_candidate[:-1]
    if base_candidate.endswith("es"):
        base_candidate = base_candidate[:-2]

    key = "morph_" + base_candidate
    if key in cache:
        return cache[key]

    for drug in short_drugs:
        # If the drug is in the base_candidate or vice versa
        # or if they share a substring of length >= 4
        if drug in base_candidate or base_candidate in drug:
            cache[key] = True
            return True
        # You could also incorporate a small edit-distance check here.

    cache[key] = False
    return False


##################################################
# Individual feature extraction steps
##################################################

def extract_basic_features(t, tokenFeatures):
    """Basic form features (same as your original approach)."""
    tokenFeatures.append("form=" + t)
    tokenFeatures.append("formLower=" + t.lower())

    if len(t) >= 3:
        tokenFeatures.append("pref3=" + t[:3].lower())
        tokenFeatures.append("suf3=" + t[-3:].lower())
    if len(t) >= 4:
        tokenFeatures.append("pref4=" + t[:4].lower())
        tokenFeatures.append("suf4=" + t[-4:].lower())
    if len(t) >= 5:
        tokenFeatures.append("pref5=" + t[:5].lower())
        tokenFeatures.append("suf5=" + t[-5:].lower())


def extract_character_features(t, tokenFeatures, chemical_patterns,
                               common_drug_prefixes, common_drug_suffixes):
    """Character-level features, including original logic plus shape."""
    # Character n-grams
    for ngram in char_ngrams(t, 2):
        tokenFeatures.append("char2gram=" + ngram)

    # Word shape
    w_shape = word_shape(t)
    tokenFeatures.append("wordShape=" + w_shape)

    # Casing
    tokenFeatures.append("casing=" + casing(t))

    # Additional checks
    if any(char.isdigit() for char in t):
        tokenFeatures.append("hasDigit=true")
    if "-" in t:
        tokenFeatures.append("hasHyphen=true")
    if "(" in t or ")" in t:
        tokenFeatures.append("hasParenthesis=true")
    if "[" in t or "]" in t:
        tokenFeatures.append("hasBracket=true")

    if has_chemical_pattern(t, chemical_patterns):
        tokenFeatures.append("hasChemicalPattern=true")
    if has_drug_affix(t, common_drug_prefixes, common_drug_suffixes):
        tokenFeatures.append("hasDrugAffix=true")

    if len(t) > 10:
        tokenFeatures.append("isLongWord=true")


def extract_context_features(tokens, k, tokenFeatures):
    """Extract local context features (previous/next tokens)."""
    if k > 0:
        tPrev = tokens[k - 1][0]
        tokenFeatures.append("formPrev=" + tPrev)
        tokenFeatures.append("formLowerPrev=" + tPrev.lower())
        tokenFeatures.append("casingPrev=" + casing(tPrev))
        # Bigram
        tokenFeatures.append("bigram=" + tPrev.lower() + "_" + tokens[k][0].lower())

        if k > 1:
            tPrev2 = tokens[k - 2][0]
            tokenFeatures.append("formPrev2=" + tPrev2)
            # Trigram
            trigram = (tPrev2.lower() + "_" + tPrev.lower() + "_" + tokens[k][0].lower())
            tokenFeatures.append("trigram=" + trigram)
    else:
        tokenFeatures.append("BoS")

    if k < len(tokens) - 1:
        tNext = tokens[k + 1][0]
        tokenFeatures.append("formNext=" + tNext)
        tokenFeatures.append("formLowerNext=" + tNext.lower())
        tokenFeatures.append("casingNext=" + casing(tNext))

        if k < len(tokens) - 2:
            tNext2 = tokens[k + 2][0]
            tokenFeatures.append("formNext2=" + tNext2)
    else:
        tokenFeatures.append("EoS")


def extract_lexicon_features(tokens, k, tokenFeatures, lexicon_data):
    """Extended dictionary-based features with basic normalizations."""
    surface1 = tokens[k][0].lower().rstrip(".,;:!?-")
    # Remove basic plural endings
    normed = surface1
    if normed.endswith("'s"):
        normed = normed[:-2]
    if normed.endswith("s"):
        normed = normed[:-1]

    # Grab windows if needed
    # (Same as original, but we can also do normed versions.)
    surface2 = (tokens[k][0] + " " + tokens[k + 1][0]).lower() if (k < len(tokens) - 1) else ""
    surface3 = ""
    if k < len(tokens) - 2:
        surface3 = (tokens[k][0] + " " + tokens[k + 1][0] + " " + tokens[k + 2][0]).lower()

    # Unpack
    drugbank_lexicon = lexicon_data['drugbank_lexicon']
    drugbank_types = lexicon_data['drugbank_types']
    hsdb_lexicon = lexicon_data['hsdb_lexicon']

    # Exact matches using normalized forms
    if normed in drugbank_lexicon:
        tokenFeatures.append("exactInDrugBank=true")
        if normed in drugbank_types:
            tokenFeatures.append("drugType=" + drugbank_types[normed])

    if normed in hsdb_lexicon:
        tokenFeatures.append("exactInHSDB=true")

    # Window-based checks (optional)
    if surface2 in drugbank_lexicon or surface3 in drugbank_lexicon:
        tokenFeatures.append("windowInDrugBank=true")
    if surface2 in hsdb_lexicon or surface3 in hsdb_lexicon:
        tokenFeatures.append("windowInHSDB=true")

    # Partial matches
    partial_match_cache_db = {}
    partial_match_cache_hsdb = {}

    if partial_match_morphological(tokens[k][0], lexicon_data['short_drugs_db'], partial_match_cache_db):
        tokenFeatures.append("partialMatchDB=true")
    if partial_match_morphological(tokens[k][0], lexicon_data['short_drugs_hsdb'], partial_match_cache_hsdb):
        tokenFeatures.append("partialMatchHSDB=true")


##################################################
# NEW: Part-of-speech tagging
##################################################

def extract_pos_features(pos_tags, index, tokenFeatures):
    """Add POS feature from the precomputed list of (token, POS)."""
    # Check index boundary and that the tag tuple/list is valid and has a non-empty tag value
    if index < len(pos_tags) and pos_tags[index] and len(pos_tags[index]) > 1 and pos_tags[index][1]:
        pos_tag_value = str(pos_tags[index][1])  # Ensure string
        tokenFeatures.append(f"pos={pos_tag_value}")
    else:
        # Handle missing or invalid POS tag, add a placeholder
        tokenFeatures.append("pos=UNK")

    ##################################################


# Master function: extract_features
##################################################

def extract_features(tokens, lexicon_data):
    """
    * tokens: list of (word, start_offset, end_offset)
    * lexicon_data: dictionary from load_lexicons() 
    Returns: list of feature lists, each a list of strings
    """
    # Precompute POS tags for entire sentence
    # NLTK pos_tag expects just token strings
    token_texts = [t[0] for t in tokens]

    # Perform POS tagging with error handling
    pos_tags = []
    try:
        if token_texts:  # Avoid tagging empty list
            pos_tags = nltk.pos_tag(token_texts)

        # Validate output length and pad if necessary
        if len(pos_tags) != len(token_texts):
            print(
                f"Warning: POS tag count ({len(pos_tags)}) != token count ({len(token_texts)}) for sentence '{' '.join(token_texts)}'. Padding with UNK.",
                file=sys.stderr)
            # Pad with placeholder tuples ('token_text', 'UNK')
            pos_tags = pos_tags + [(token_texts[i] if i < len(token_texts) else 'UNK', 'UNK') for i in
                                   range(len(pos_tags), len(token_texts))]

    except Exception as e:
        print(f"Error during NLTK POS tagging for sentence '{' '.join(token_texts)}': {e}. Using UNK tags.",
              file=sys.stderr)
        # Create placeholder tags if tagging fails completely
        pos_tags = [(text, 'UNK') for text in token_texts]

    all_features = []
    for k in range(len(tokens)):
        t = tokens[k][0]
        tokenFeatures = []

        # Basic
        extract_basic_features(t, tokenFeatures)
        # Character-level & shape
        extract_character_features(t, tokenFeatures,
                                   lexicon_data['chemical_patterns'],
                                   lexicon_data['common_drug_prefixes'],
                                   lexicon_data['common_drug_suffixes'])
        # Context
        extract_context_features(tokens, k, tokenFeatures)
        # Dictionary-based
        extract_lexicon_features(tokens, k, tokenFeatures, lexicon_data)
        # Part-of-speech
        extract_pos_features(pos_tags, k, tokenFeatures)

        all_features.append(tokenFeatures)

    return all_features
