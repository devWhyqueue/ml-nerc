#!/usr/bin/python3
"""Enhanced feature extraction for a CRF-based NER system.
Adds: POS tagging (NLTK), word shape features, expanded partial match logic, improved lexicon usage.
"""
import sys

import nltk
from rapidfuzz import process, fuzz

from embeddings import get_embedding_features


def casing(token):
    """Determine token casing pattern."""
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
    """Extract limited character n-grams (max 5)."""
    if len(token) < n:
        return [token.lower()]
    return [token[i:i + n].lower() for i in range(min(5, len(token) - n + 1))]

def word_shape(token):
    """Return a simplified shape of the token (e.g., 'Acetaminophen' -> 'Xxxxxxxxxxxxx')."""
    shape = []
    for char in token:
        if char.isdigit():
            shape.append("d")
        elif char.isalpha():
            shape.append("X" if char.isupper() else "x")
        else:
            shape.append(char)
    return "".join(shape)


def extract_basic_features(t, feats):
    """Add basic form features."""
    feats += ["form=" + t, "formLower=" + t.lower()]
    if len(t) >= 3:
        feats += ["pref3=" + t[:3].lower(), "suf3=" + t[-3:].lower()]
    if len(t) >= 4:
        feats += ["pref4=" + t[:4].lower(), "suf4=" + t[-4:].lower()]
    if len(t) >= 5:
        feats += ["pref5=" + t[:5].lower(), "suf5=" + t[-5:].lower()]


def extract_character_features(t, feats):
    """Add character-level features including n-grams, word shape and casing."""
    for ng in char_ngrams(t, 2):
        feats.append("char2gram=" + ng)
    feats.append("wordShape=" + word_shape(t))
    feats.append("casing=" + casing(t))
    if any(c.isdigit() for c in t): feats.append("hasDigit=true")
    if "-" in t: feats.append("hasHyphen=true")
    if "(" in t or ")" in t: feats.append("hasParenthesis=true")
    if "[" in t or "]" in t: feats.append("hasBracket=true")
    if len(t) > 10: feats.append("isLongWord=true")


def extract_context_features(tokens, k, feats):
    """Add context features (previous/next tokens and n-gram context)."""
    if k > 0:
        tPrev = tokens[k - 1][0]
        feats += ["formPrev=" + tPrev, "formLowerPrev=" + tPrev.lower(), "casingPrev=" + casing(tPrev),
                  "bigram=" + tPrev.lower() + "_" + tokens[k][0].lower()]
        if k > 1:
            tPrev2 = tokens[k - 2][0]
            feats.append("formPrev2=" + tPrev2)
            feats.append("trigram=" + tPrev2.lower() + "_" + tPrev.lower() + "_" + tokens[k][0].lower())
    else:
        feats.append("BoS")
    if k < len(tokens) - 1:
        tNext = tokens[k + 1][0]
        feats += ["formNext=" + tNext, "formLowerNext=" + tNext.lower(), "casingNext=" + casing(tNext)]
        if k < len(tokens) - 2:
            feats.append("formNext2=" + tokens[k + 2][0])
    else:
        feats.append("EoS")


_fuzzy_cache = {}


def fuzzy_lexicon_match(token, lexicon_list, cutoff=85):
    global _fuzzy_cache
    t_norm = token.lower().strip()
    cache_key = (t_norm, id(lexicon_list))
    if cache_key in _fuzzy_cache:
        return _fuzzy_cache[cache_key]
    match = process.extractOne(t_norm, lexicon_list, scorer=fuzz.ratio, score_cutoff=cutoff)
    result = match is not None
    _fuzzy_cache[cache_key] = result
    return result


def extract_lexicon_features(tokens, k, feats, lexicon_data):
    token = tokens[k][0]
    # Use rapidfuzz fuzzy matching with cache
    if fuzzy_lexicon_match(token, lexicon_data['approved_drugs_lexicon'], cutoff=85):
        feats.append('fuzzyApprovedDrugsLexicon=true')
    if fuzzy_lexicon_match(token, lexicon_data['not_approved_drugs_lexicon'], cutoff=85):
        feats.append('fuzzyNotApprovedDrugsLexicon=true')


def extract_pos_features(pos_tags, idx, feats):
    """Add POS tag features from the precomputed list."""
    if idx < len(pos_tags) and pos_tags[idx] and len(pos_tags[idx]) > 1 and pos_tags[idx][1]:
        feats.append("pos=" + str(pos_tags[idx][1]))
    else:
        feats.append("pos=UNK")


def extract_features(tokens, lexicon_data):
    """
    tokens: list of (word, start_offset, end_offset)
    lexicon_data: dictionary from load_lexicons()
    Returns a list of feature lists (each a list of feature strings).
    """
    token_texts = [t[0] for t in tokens]
    try:
        pos_tags = nltk.pos_tag(token_texts) if token_texts else []
        if len(pos_tags) != len(token_texts):
            print(f"Warning: POS tag count ({len(pos_tags)}) != token count ({len(token_texts)}) for sentence "
                  f"'{' '.join(token_texts)}'. Padding with UNK.", file=sys.stderr)
            pos_tags += [(token_texts[i] if i < len(token_texts) else 'UNK', 'UNK')
                         for i in range(len(pos_tags), len(token_texts))]
    except Exception as e:
        print(f"Error during NLTK POS tagging for sentence "
              f"'{' '.join(token_texts)}': {e}. Using UNK tags.", file=sys.stderr)
        pos_tags = [(text, 'UNK') for text in token_texts]
    all_feats = []
    for k in range(len(tokens)):
        t = tokens[k][0]
        feats = []
        extract_basic_features(t, feats)
        extract_character_features(t, feats)
        extract_context_features(tokens, k, feats)
        extract_pos_features(pos_tags, k, feats)
        emb_feats = get_embedding_features(t, lexicon_data['word_embeddings'])
        feats.extend(emb_feats)
        extract_lexicon_features(tokens, k, feats, lexicon_data)

        all_feats.append(feats)
    return all_feats
