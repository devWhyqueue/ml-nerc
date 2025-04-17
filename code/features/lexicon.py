#!/usr/bin/python3

def load_lexicons():
    """Load approved and not approved drug lexicons from files."""
    approved_drugs_lexicon = load_approved_drugs()
    not_approved_drugs_lexicon = load_not_approved_drugs()
    return {
        'approved_drugs_lexicon': approved_drugs_lexicon,
        'not_approved_drugs_lexicon': not_approved_drugs_lexicon,
    }


def load_approved_drugs():
    """Load unique approved drug names from the combined file."""
    with open("../data/lexicon/approved_drugs.txt", encoding="utf-8") as f:
        approved_drugs_lexicon = set(line.strip().lower() for line in f if line.strip())
    return approved_drugs_lexicon


def load_not_approved_drugs():
    """Load unique not approved drug names from the file."""
    with open("../data/lexicon/not_approved_drugs.txt", encoding="utf-8") as f:
        not_approved_drugs_lexicon = set(line.strip().lower() for line in f if line.strip())
    return not_approved_drugs_lexicon
