#! /bin/bash

BASEDIR=..
OUTPUT_DIR="$BASEDIR/.output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# convert datasets to feature vectors
echo "Extracting features..."
python features/main.py $BASEDIR/data/train/ > "$OUTPUT_DIR/train.feat"
python features/main.py $BASEDIR/data/devel/ > "$OUTPUT_DIR/devel.feat"

# train CRF model
echo "Training CRF model..."
python train-crf.py "$OUTPUT_DIR/model.crf" < "$OUTPUT_DIR/train.feat"
# run CRF model
echo "Running CRF model..."
python predict.py "$OUTPUT_DIR/model.crf" < "$OUTPUT_DIR/devel.feat" > "$OUTPUT_DIR/devel-CRF.out"
python predict.py "$OUTPUT_DIR/model.crf" < "$OUTPUT_DIR/train.feat" > "$OUTPUT_DIR/train-CRF.out"
# evaluate CRF results
echo "Evaluating CRF results..."
python evaluator.py NER $BASEDIR/data/train "$OUTPUT_DIR/train-CRF.out" > "$OUTPUT_DIR/train-CRF.stats"
python evaluator.py NER $BASEDIR/data/devel "$OUTPUT_DIR/devel-CRF.out" > "$OUTPUT_DIR/devel-CRF.stats"


#Extract Classification Features
cat "$OUTPUT_DIR/train.feat" | cut -f5- | grep -v ^$ > "$OUTPUT_DIR/train.clf.feat"


# train Naive Bayes model
echo "Training Naive Bayes model..."
python train-sklearn.py "$OUTPUT_DIR/model.joblib" "$OUTPUT_DIR/vectorizer.joblib" < "$OUTPUT_DIR/train.clf.feat"
# run Naive Bayes model
echo "Running Naive Bayes model..."
python predict-sklearn.py "$OUTPUT_DIR/model.joblib" "$OUTPUT_DIR/vectorizer.joblib" < "$OUTPUT_DIR/devel.feat" > "$OUTPUT_DIR/devel-NB.out"
# evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
python evaluator.py NER $BASEDIR/data/devel "$OUTPUT_DIR/devel-NB.out" > "$OUTPUT_DIR/devel-NB.stats"

# remove auxiliary files.
rm "$OUTPUT_DIR/train.clf.feat"
