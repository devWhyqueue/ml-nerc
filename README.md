# Named Entity Recognition and Classification (NERC) for Drug Names

This repository contains the LaTeX report and associated files for the Named Entity Recognition and Classification (NERC) project focused on drug names in biomedical text. The project was completed as part of the Mining Unstructured Data course.

## Project Overview

The project implements a machine learning approach to NERC for drug names using the DDI (Drug-Drug Interaction) corpus. The task involves identifying mentions of pharmacological substances and classifying them into four categories:
- drug (generic drug names)
- brand (brand or trade names)
- group (pharmacological classes or categories of drugs)
- drug_n (active substances not approved for human use)

We implement a sequence tagging model using the B-I-O encoding scheme and a Conditional Random Field (CRF) classifier to jointly model token-level features and label transitions. We also compare the CRF performance with a baseline Naïve Bayes classifier.

## Repository Structure

```
.
├── README.md                  # This file
├── code/                      # Python implementation
│   ├── codemaps.py            # Feature extraction and mapping utilities
│   ├── dataset.py             # Dataset loading and processing
│   ├── predict.py             # Model prediction script
│   ├── run.sh                 # Shell script to run the pipeline
│   └── train.py               # Model training script
├── data/                      # DDI corpus data
│   ├── train/                 # Training set XML files
│   ├── devel/                 # Development set XML files
│   └── test/                  # Test set XML files
├── report/                    # LaTeX report files
└── task/                      # Task description
```

## Key Features

The NERC system implements several key features:
- Rich feature engineering including lexical form, orthographic patterns, and contextual cues
- Integration of external domain knowledge through lexicons of known drug names
- Comparison between CRF and Naïve Bayes approaches
- Detailed analysis of feature impact through ablation studies
- Error analysis and insights for future improvements

## Setup and Installation

To set up a virtual environment and install the required dependencies:

1. Make sure you have Python installed (Python 3.8+ recommended)
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
4. Install the dependencies from requirements.txt:
   ```
   pip install -r requirements.txt
   ```

Once the dependencies are installed, you can run the code as described in the code directory.

## Results

Our CRF-based model achieves a macro-F1 score of approximately 72% on the test set, approaching the performance of the best systems in the SemEval-2013 DDI challenge. The ablation studies reveal that lexicon features and contextual information are particularly valuable for drug name recognition.

## References

The report cites several key references, including:
- The DDI corpus paper
- The SemEval-2013 DDI extraction challenge
- CRF methodology papers
- Scikit-learn documentation

For the complete list of references, see the `references.bib` file.