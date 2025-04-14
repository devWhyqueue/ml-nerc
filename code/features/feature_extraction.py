#!/usr/bin/python3
"""Enhanced feature extraction for a CRF-based NER system.
Adds: POS tagging (NLTK), word shape features, expanded partial match logic, improved lexicon usage.
"""
import sys

import nltk

from drug_n_features import extract_drug_n_features
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


def has_chemical_pattern(token, chemical_patterns):
    """Return True if token matches any chemical regex pattern."""
    return any(pattern.search(token) for pattern in chemical_patterns)


def has_drug_affix(token, common_drug_prefixes, common_drug_suffixes):
    """Return True if token has drug-like prefix or suffix."""
    t = token.lower()
    if len(t) < 3: return False
    return (t[:3] in common_drug_prefixes or t[-3:] in common_drug_suffixes)


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


def partial_match_morphological(token, short_drugs, cache):
    """Extended partial match: try simple morphological variants (removing trailing plural suffixes)."""
    base = token.lower().rstrip(".,;:!?-").replace("'s", "")
    if base.endswith("es"):
        base = base[:-2]
    elif base.endswith("s"):
        base = base[:-1]
    key = "morph_" + base
    if key in cache: return cache[key]
    for drug in short_drugs:
        if drug in base or base in drug:
            cache[key] = True
            return True
    cache[key] = False
    return False


def extract_basic_features(t, feats):
    """Add basic form features."""
    feats += ["form=" + t, "formLower=" + t.lower()]
    if len(t) >= 3:
        feats += ["pref3=" + t[:3].lower(), "suf3=" + t[-3:].lower()]
    if len(t) >= 4:
        feats += ["pref4=" + t[:4].lower(), "suf4=" + t[-4:].lower()]
    if len(t) >= 5:
        feats += ["pref5=" + t[:5].lower(), "suf5=" + t[-5:].lower()]


def extract_character_features(t, feats, chemical_patterns, common_drug_prefixes, common_drug_suffixes):
    """Add character-level features including n-grams, word shape and casing."""
    for ng in char_ngrams(t, 2):
        feats.append("char2gram=" + ng)
    feats.append("wordShape=" + word_shape(t))
    feats.append("casing=" + casing(t))
    if any(c.isdigit() for c in t): feats.append("hasDigit=true")
    if "-" in t: feats.append("hasHyphen=true")
    if "(" in t or ")" in t: feats.append("hasParenthesis=true")
    if "[" in t or "]" in t: feats.append("hasBracket=true")
    if has_chemical_pattern(t, chemical_patterns): feats.append("hasChemicalPattern=true")
    if has_drug_affix(t, common_drug_prefixes, common_drug_suffixes): feats.append("hasDrugAffix=true")
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


def extract_lexicon_features(tokens, k, feats, lexicon_data):
    """Add features based on lexicon matches with normalization and window checks."""
    surface1 = tokens[k][0].lower().rstrip(".,;:!?-")
    normed = surface1
    if normed.endswith("'s"):
        normed = normed[:-2]
    elif normed.endswith("s"):
        normed = normed[:-1]
    surface2 = (tokens[k][0] + " " + tokens[k + 1][0]).lower() if k < len(tokens) - 1 else ""
    surface3 = (tokens[k][0] + " " + tokens[k + 1][0] + " " + tokens[k + 2][0]).lower() if k < len(tokens) - 2 else ""
    drugbank_lex = lexicon_data['drugbank_lexicon']
    drugbank_types = lexicon_data['drugbank_types']
    hsdb_lex = lexicon_data['hsdb_lexicon']
    if normed in drugbank_lex:
        feats.append("exactInDrugBank=true")
        if normed in drugbank_types:
            feats.append("drugType=" + drugbank_types[normed])
    if normed in hsdb_lex: feats.append("exactInHSDB=true")
    if surface2 in drugbank_lex or surface3 in drugbank_lex: feats.append("windowInDrugBank=true")
    if surface2 in hsdb_lex or surface3 in hsdb_lex: feats.append("windowInHSDB=true")
    cache_db, cache_hsdb = {}, {}
    if partial_match_morphological(tokens[k][0], lexicon_data['short_drugs_db'], cache_db):
        feats.append("partialMatchDB=true")
    if partial_match_morphological(tokens[k][0], lexicon_data['short_drugs_hsdb'], cache_hsdb):
        feats.append("partialMatchHSDB=true")


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
        extract_character_features(t, feats, lexicon_data['chemical_patterns'],
                                   lexicon_data['common_drug_prefixes'],
                                   lexicon_data['common_drug_suffixes'])
        extract_context_features(tokens, k, feats)
        extract_lexicon_features(tokens, k, feats, lexicon_data)
        extract_pos_features(pos_tags, k, feats)

        # Add drug_n specific features if available
        if 'drug_n_lexicon' in lexicon_data and 'drug_n_patterns' in lexicon_data:
            drug_n_feats = extract_drug_n_features(t,
                                                   lexicon_data['drug_n_lexicon'],
                                                   lexicon_data['drug_n_patterns'])
            feats.extend(drug_n_feats)

        # Add word embedding features if available
        if 'word_embeddings' in lexicon_data:
            emb_feats = get_embedding_features(t, lexicon_data['word_embeddings'])
            feats.extend(emb_feats)
            
        all_feats.append(feats)
    return all_feats
