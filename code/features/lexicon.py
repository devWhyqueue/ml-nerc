#!/usr/bin/python3

import re


# Lexicon loading and preparation functions
def load_lexicons():
    """Load drug lexicons from files and prepare related data structures."""
    drugbank_lexicon, drugbank_types = load_drugbank()
    hsdb_lexicon = load_hsdb()

    # Create prefix and suffix sets from the lexicons for better matching
    common_drug_prefixes, common_drug_suffixes = extract_affixes(drugbank_lexicon)

    # Store a sample of shorter drugs for partial matching
    short_drugs_db = sample_short_drugs(drugbank_lexicon, 500)
    short_drugs_hsdb = sample_short_drugs(hsdb_lexicon, 200)

    # Precompile regex patterns for better performance
    chemical_patterns = compile_chemical_patterns()

    return {
        'drugbank_lexicon': drugbank_lexicon,
        'drugbank_types': drugbank_types,
        'hsdb_lexicon': hsdb_lexicon,
        'common_drug_prefixes': common_drug_prefixes,
        'common_drug_suffixes': common_drug_suffixes,
        'chemical_patterns': chemical_patterns,
        'short_drugs_db': short_drugs_db,
        'short_drugs_hsdb': short_drugs_hsdb
    }


def load_drugbank():
    """Load DrugBank lexicon and extract drug types."""
    with open("../data/lexicon/DrugBank.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        drugbank_lexicon = set(line.split('|')[0].lower() for line in lines if line)

        # Extract drug types from DrugBank
        drugbank_types = {}
        for line in lines:
            parts = line.split('|')
            if len(parts) > 1:
                drugbank_types[parts[0].lower()] = parts[1]

    return drugbank_lexicon, drugbank_types


def load_hsdb():
    """Load HSDB lexicon."""
    with open("../data/lexicon/HSDB.txt", encoding="utf-8") as f:
        hsdb_lexicon = set(line.strip().lower() for line in f if line.strip())

    return hsdb_lexicon


def extract_affixes(drug_lexicon):
    """Extract common prefixes and suffixes from drug names."""
    common_drug_prefixes = set()
    common_drug_suffixes = set()

    for drug in drug_lexicon:
        if len(drug) > 3:
            common_drug_prefixes.add(drug[:3].lower())
            common_drug_suffixes.add(drug[-3:].lower())

    return common_drug_prefixes, common_drug_suffixes


def sample_short_drugs(lexicon, sample_size):
    """Sample a subset of shorter drugs for partial matching."""
    return sorted([drug for drug in lexicon if 4 < len(drug) < 10])[:sample_size]


def compile_chemical_patterns():
    """Compile regex patterns for chemical entity recognition."""
    return [
        re.compile(r'\d+[,.-]\d+'),  # Numbers with separators
        re.compile(r'[A-Za-z]\d'),  # Letter followed by number
        re.compile(r'\d[A-Za-z]'),  # Number followed by letter
        re.compile(r'[A-Za-z]-\d'),  # Letter-dash-number
        re.compile(r'\d-[A-Za-z]'),  # Number-dash-letter
        re.compile(r'[A-Za-z]\([A-Za-z0-9]+\)'),  # Chemical formulas
        re.compile(r'[A-Za-z]\[[A-Za-z0-9]+\]'),  # Chemical notations
    ]
