#!/usr/bin/python3

import argparse
import os
import random
import sys
from os import listdir, path
from xml.dom.minidom import parse

from tqdm import tqdm

from embeddings import load_embeddings
from feature_extraction import extract_features
from lexicon import load_lexicons
from tokenization import tokenize, get_tag

# Set random seed for reproducibility
random.seed(42)


def load_resources():
    """Load lexicons and word embeddings if available."""
    # Load lexicons (expects data relative to lexicon.py)
    lexicon_data = load_lexicons()

    # Try to load word embeddings if available
    EMBEDDING_FILE = os.environ.get("EMBEDDING_FILE", "../data/embeddings/glove.6B.50d.txt")
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))

    # Check if embedding file exists before attempting to load
    if os.path.exists(EMBEDDING_FILE):
        print(f"Loading word embeddings from {EMBEDDING_FILE}...", file=sys.stderr)
        lexicon_data['word_embeddings'] = load_embeddings(EMBEDDING_FILE, EMBEDDING_DIM)
    else:
        print(f"Warning: Embedding file {EMBEDDING_FILE} not found. Word embedding features will not be used.",
              file=sys.stderr)
        print(
            f"To use embeddings, download a pre-trained model (e.g., GloVe) and set the EMBEDDING_FILE environment variable.",
            file=sys.stderr)
        lexicon_data['word_embeddings'] = {}

    return lexicon_data


def process_file(filepath, outfile, lexicon_data):
    """Process a single XML file and write features to the output file."""
    # Parse XML file, obtaining a DOM tree
    tree = parse(filepath)

    # Process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text

        # Basic SID validation
        if not sid or sid == "\\n":
            print(f"WARNING: Skipping sentence due to invalid sid: {repr(sid)}", file=sys.stderr)
            continue

        spans = []

        # Extract entity spans
        entities = s.getElementsByTagName("entity")
        for e in entities:
            char_offsets = e.attributes["charOffset"].value.split(";")
            typ = e.attributes["type"].value

            for offset in char_offsets:
                start, end = offset.split("-")
                spans.append((int(start), int(end), typ))

        # Convert the sentence to a list of tokens
        tokens = tokenize(stext)

        # Extract sentence features
        features = extract_features(tokens, lexicon_data)

        # Print features in format expected by crfsuite trainer
        for i in range(0, len(tokens)):
            token_text = tokens[i][0]
            # Skip empty tokens
            if not token_text:
                continue

            # Ensure token tuple structure is correct
            if len(tokens[i]) != 3:
                print(f"WARNING (sid={sid}): Malformed token tuple structure. Skipping.", file=sys.stderr)
                continue

            token_start = tokens[i][1]
            token_end = tokens[i][2]
            tag = get_tag(tokens[i], spans)

            # Basic validation
            if not tag:
                continue

            # Safety check for features
            try:
                current_features = features[i]
            except IndexError:
                print(f"WARNING (sid={sid}): Feature list index out of range. Skipping token.", file=sys.stderr)
                continue

            # Sanitize features and token
            sanitized_features = [str(f).replace('\n', '\\n').replace('\t', '\\t') for f in current_features]
            sanitized_token_text = token_text.replace('\n', '\\n').replace('\t', '\\t')

            # Write features for CRFsuite
            try:
                feature_str = '\t'.join(sanitized_features)
                outfile.write(f"{sid}\t{sanitized_token_text}\t{token_start}\t{token_end}\t{tag}\t{feature_str}\n")
            except Exception as e:
                print(f"ERROR (sid={sid}): Failed to write output line: {e}", file=sys.stderr)
                continue

        # Write blank line to separate sentences
        outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Extract features from XML files for CRFsuite.")
    parser.add_argument("input_dir", help="Directory containing the input XML files.")
    args = parser.parse_args()

    try:
        # Load resources only when the script is executed directly
        lexicon_data = load_resources()
        
        # Process each file in the input directory and write to stdout
        for filename in tqdm(listdir(args.input_dir), desc="Extracting features"):
            if filename.endswith(".xml"):
                filepath = path.join(args.input_dir, filename)
                if path.isfile(filepath):  # Ensure it's a file
                    process_file(filepath, sys.stdout, lexicon_data)  # Pass lexicon_data as argument
        print(f"Feature extraction complete.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
