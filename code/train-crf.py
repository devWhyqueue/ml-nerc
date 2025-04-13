#!/usr/bin/env python3

import sys
from collections import Counter

import pycrfsuite


def instances(fi):
    xseq = []
    yseq = []
    
    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line means the end of a sentence.
            # Return accumulated sequences, and reinitialize.
            yield xseq, yseq
            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')
        
        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]        
        xseq.append(item)
        
        # Append the label to the label sequence.
        yseq.append(fields[4])


def calculate_class_weights(all_labels):
    """
    Calculate weights for each class based on their frequency.
    Uses a more sophisticated approach for rare classes.
    """
    # Count occurrences of each label
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)

    # Get only the entity labels (excluding 'O')
    entity_labels = {label: count for label, count in label_counts.items() if label != 'O'}

    # Calculate weights
    weights = {}

    # Set weight for 'O' (background) class to 1.0
    weights['O'] = 1.0

    # For entity classes, calculate weights based on inverse frequency
    if entity_labels:
        # Find the most common entity label
        most_common_entity = max(entity_labels.items(), key=lambda x: x[1])
        most_common_entity_count = most_common_entity[1]

        # Calculate weights for entity classes
        for label, count in entity_labels.items():
            # Base weight is inverse of frequency relative to most common entity
            base_weight = most_common_entity_count / count

            # Apply special boosting for very rare classes (especially drug_n)
            if 'drug_n' in label:
                # Extra boost for drug_n class
                weights[label] = base_weight * 5.0
            elif count < most_common_entity_count / 10:
                # Boost other rare classes
                weights[label] = base_weight * 2.0
            else:
                weights[label] = base_weight

    return weights


if __name__ == '__main__':
    # get file where model will be written
    modelfile = sys.argv[1]

    # Create a Trainer object
    trainer = pycrfsuite.Trainer(verbose=True)

    # First pass: collect all labels to calculate class weights
    all_labels = []
    training_data = []
    
    for xseq, yseq in instances(sys.stdin):
        all_labels.extend(yseq)
        training_data.append((xseq, yseq))

    # Calculate weights for each class
    class_weights = calculate_class_weights(all_labels)

    # Print class distribution and weights
    label_counts = Counter(all_labels)
    print(f"Class distribution: {label_counts}", file=sys.stderr)
    print(f"Total samples: {len(all_labels)}", file=sys.stderr)
    print(f"Class weights: {class_weights}", file=sys.stderr)

    # Second pass: append instances with appropriate weights
    for xseq, yseq in training_data:
        # Calculate the weight for this sequence based on its labels
        # Use maximum weight to prioritize sequences with rare classes
        weights = [class_weights[y] for y in yseq]
        sequence_weight = max(weights) if weights else 1.0

        trainer.append(xseq, yseq, sequence_weight)

    # Use L2-regularized SGD and 1st-order dyad features
    trainer.select('lbfgs', 'crf1d')

    # Set basic parameters
    trainer.set('feature.minfreq', 1)
    trainer.set('c2', 0.001)

    # Increase max iterations for better convergence with imbalanced data
    trainer.set('max_iterations', 300)
    trainer.set('epsilon', 1e-5)

    print("Training with following parameters: ", file=sys.stderr)
    for name in trainer.params():
        print(f"{name}: {trainer.get(name)} - {trainer.help(name)}", file=sys.stderr)
    
    # Start training and dump model to modelfile
    trainer.train(modelfile, -1)
