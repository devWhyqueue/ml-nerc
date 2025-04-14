#!/usr/bin/python3
"""
Domain-specific features for drug_n (non-proprietary drug names) recognition.
Provides specialized lexicon checks and pattern matching for drug_n entities.
"""
import re
import sys


def load_drug_n_lexicon(filepath=None):
    """
    Load a specialized lexicon for non-proprietary drug names.
    
    Args:
        filepath: Path to the drug_n lexicon file
        
    Returns:
        Set of non-proprietary drug names
    """
    drug_n_lexicon = set()

    # Default to an empty set if no file is provided
    if not filepath:
        return drug_n_lexicon

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            drug_n_lexicon = set(line.strip().lower() for line in f if line.strip())
        print(f"Loaded {len(drug_n_lexicon)} non-proprietary drug names", file=sys.stderr)
    except Exception as e:
        print(f"Error loading drug_n lexicon: {e}", file=sys.stderr)

    return drug_n_lexicon


def compile_drug_n_patterns():
    """
    Compile regex patterns specific to non-proprietary drug names.
    
    Returns:
        List of compiled regex patterns
    """
    return [
        # Common non-proprietary drug name suffixes
        re.compile(
            r'(?i)(in|ol|ine|ide|ate|ium|one|ase|amine|amide|azole|icin|mycin|oxacin|cillin|cycline|dipine|pril|sartan|statin|zepam|zolam|prazole|oxetine|azepam)$'),

        # Common patterns in non-proprietary names
        re.compile(r'(?i)[a-z]{4,}(acid|salt)$'),

        # INN (International Nonproprietary Names) stem patterns
        re.compile(r'(?i)(cef|dopa|mab|nib|pril|sartan|tinib|vastatin|prazole)'),

        # Chemical compound patterns common in non-proprietary names
        re.compile(r'(?i)[a-z]+(sodium|potassium|calcium|magnesium|chloride|sulfate|phosphate)$'),
    ]


def partial_match_drug_n(token, drug_n_lexicon, drug_n_patterns=None):
    """
    Check if a token matches patterns typical of non-proprietary drug names.
    
    Args:
        token: The token to check
        drug_n_lexicon: Set of known non-proprietary drug names
        drug_n_patterns: List of regex patterns for non-proprietary drug names
        
    Returns:
        Boolean indicating whether the token matches drug_n patterns
    """
    if not drug_n_patterns:
        drug_n_patterns = compile_drug_n_patterns()

    # Normalize token
    t_lower = token.lower().strip(".,;:!?-")

    # 1. Check exact match in lexicon
    if t_lower in drug_n_lexicon:
        return True

    # 2. Check for partial matches in lexicon (if token is long enough)
    if len(t_lower) >= 5:
        for drug in drug_n_lexicon:
            # Check if the token is a substantial substring of a known drug_n
            # or if a known drug_n is a substantial substring of the token
            if (len(drug) >= 5 and
                    (t_lower in drug or drug in t_lower) and
                    (min(len(drug), len(t_lower)) / max(len(drug), len(t_lower)) >= 0.7)):
                return True

    # 3. Check for typical drug_n patterns
    for pattern in drug_n_patterns:
        if pattern.search(token):
            return True

    # 4. Check for morphological variants
    stem = t_lower
    if stem.endswith('s'):
        stem = stem[:-1]
    if stem.endswith('e'):
        stem = stem[:-1]

    for drug in drug_n_lexicon:
        if drug.startswith(stem) and len(stem) >= 5:
            return True

    return False


def extract_drug_n_features(token, drug_n_lexicon, drug_n_patterns):
    """
    Extract features specific to non-proprietary drug name recognition.
    
    Args:
        token: The token to extract features for
        drug_n_lexicon: Set of known non-proprietary drug names
        drug_n_patterns: List of regex patterns for non-proprietary drug names
        
    Returns:
        List of drug_n specific features
    """
    features = []

    # Check for exact match in drug_n lexicon
    t_lower = token.lower()
    if t_lower in drug_n_lexicon:
        features.append("exactDrugNMatch=true")

    # Check for partial match using specialized function
    if partial_match_drug_n(token, drug_n_lexicon, drug_n_patterns):
        features.append("partialDrugNMatch=true")

    # Check for specific drug_n patterns
    for i, pattern in enumerate(drug_n_patterns):
        if pattern.search(token):
            features.append(f"drugNPattern{i}=true")

    # Check for common non-proprietary name suffixes (more specific than the patterns)
    common_suffixes = ['in', 'ol', 'ine', 'ide', 'ate', 'ium', 'one', 'ase',
                       'amine', 'amide', 'azole', 'icin', 'mycin', 'oxacin',
                       'cillin', 'cycline', 'dipine', 'pril', 'sartan', 'statin']

    t_lower = token.lower()
    for suffix in common_suffixes:
        if t_lower.endswith(suffix) and len(t_lower) > len(suffix) + 2:
            features.append(f"drugNSuffix_{suffix}=true")

    return features
