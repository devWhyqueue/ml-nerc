#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

# Lexicon loading
with open("../data/lexicon/DrugBank.txt", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    drugbank_lexicon = set(line.split('|')[0].lower() for line in lines if line)
    # Extract drug types from DrugBank
    drugbank_types = {}
    for line in lines:
        parts = line.split('|')
        if len(parts) > 1:
            drugbank_types[parts[0].lower()] = parts[1]

with open("../data/lexicon/HSDB.txt", encoding="utf-8") as f:
    hsdb_lexicon = set(line.strip().lower() for line in f if line.strip())

# Create prefix and suffix sets from the lexicons for better matching
common_drug_prefixes = set()
common_drug_suffixes = set()
for drug in drugbank_lexicon:
    if len(drug) > 3:
        common_drug_prefixes.add(drug[:3].lower())
        common_drug_suffixes.add(drug[-3:].lower())

# Precompile regex patterns for better performance
chemical_patterns = [
    re.compile(r'\d+[,.-]\d+'),  # Numbers with separators
    re.compile(r'[A-Za-z]\d'),    # Letter followed by number
    re.compile(r'\d[A-Za-z]'),    # Number followed by letter
    re.compile(r'[A-Za-z]-\d'),   # Letter-dash-number
    re.compile(r'\d-[A-Za-z]'),   # Number-dash-letter
    re.compile(r'[A-Za-z]\([A-Za-z0-9]+\)'),  # Chemical formulas
    re.compile(r'[A-Za-z]\[[A-Za-z0-9]+\]'),  # Chemical notations
]

# Create a more efficient substring lookup structure
# Store a sample of shorter drugs for partial matching
short_drugs_db = [drug for drug in drugbank_lexicon if 4 < len(drug) < 10][:500]
short_drugs_hsdb = [drug for drug in hsdb_lexicon if 4 < len(drug) < 10][:200]

   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens):
    result = []

    def casing(token):
        if token.isupper():
            return "ALLCAPS"
        elif token.istitle():
            return "Title"
        elif token.islower():
            return "LOWER"
        elif any(c.isupper() for c in token):
            return "MIXED"
        else:
            return "NOCASE"
    
    # Check if token contains chemical patterns
    def has_chemical_pattern(token):
        return any(pattern.search(token) for pattern in chemical_patterns)

    # Helper to create surface text around current token
    def window_surface(tokens, k, size):
        forms = [tokens[i][0] for i in range(k, min(k+size, len(tokens)))]
        return " ".join(forms).lower()
    
    # Helper to get backward window
    def backward_window(tokens, k, size):
        if k < size:
            return " ".join([tokens[i][0].lower() for i in range(0, k)])
        else:
            return " ".join([tokens[i][0].lower() for i in range(k-size, k)])
    
    # Helper to check if token has drug-like prefix or suffix
    def has_drug_affix(token):
        if len(token) < 3:
            return False
        return (token[:3].lower() in common_drug_prefixes or 
                token[-3:].lower() in common_drug_suffixes)
    
    # Helper to extract character n-grams - limit to 5 for performance
    def char_ngrams(token, n):
        if len(token) < n:
            return [token.lower()]
        ngrams = []
        for i in range(min(5, len(token)-n+1)):
            ngrams.append(token[i:i+n].lower())
        return ngrams

    # Cache for partial matches to avoid repeated lookups
    partial_match_cache_db = {}
    partial_match_cache_hsdb = {}
    
    # Efficient partial match check
    def has_partial_match_db(surface):
        if surface in partial_match_cache_db:
            return partial_match_cache_db[surface]
        
        # Only check against a limited set of drugs for performance
        for drug in short_drugs_db:
            if drug in surface or surface in drug:
                partial_match_cache_db[surface] = True
                return True
        
        partial_match_cache_db[surface] = False
        return False
    
    def has_partial_match_hsdb(surface):
        if surface in partial_match_cache_hsdb:
            return partial_match_cache_hsdb[surface]
        
        # Only check against a limited set of drugs for performance
        for drug in short_drugs_hsdb:
            if drug in surface or surface in drug:
                partial_match_cache_hsdb[surface] = True
                return True
        
        partial_match_cache_hsdb[surface] = False
        return False

    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        # Basic features
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("formLower=" + t.lower())
        
        # Enhanced suffix/prefix features - only add if length is sufficient
        if len(t) >= 3:
            tokenFeatures.append("pref3=" + t[:3].lower())
            tokenFeatures.append("suf3=" + t[-3:].lower())
        if len(t) >= 4:
            tokenFeatures.append("pref4=" + t[:4].lower())
            tokenFeatures.append("suf4=" + t[-4:].lower())
        if len(t) >= 5:
            tokenFeatures.append("pref5=" + t[:5].lower())
            tokenFeatures.append("suf5=" + t[-5:].lower())
            
        # Character-level n-grams - limited to first 5 for performance
        for ngram in char_ngrams(t, 2):
            tokenFeatures.append("char2gram=" + ngram)
        
        # Casing feature
        tokenFeatures.append("casing=" + casing(t))

        # Character pattern features
        if any(char.isdigit() for char in t):
            tokenFeatures.append("hasDigit=true")
        if "-" in t:
            tokenFeatures.append("hasHyphen=true")
        if "(" in t or ")" in t:
            tokenFeatures.append("hasParenthesis=true")
        if "[" in t or "]" in t:
            tokenFeatures.append("hasBracket=true")
        if has_chemical_pattern(t):
            tokenFeatures.append("hasChemicalPattern=true")
        if has_drug_affix(t):
            tokenFeatures.append("hasDrugAffix=true")
        
        # Length features
        if len(t) > 10:
            tokenFeatures.append("isLongWord=true")
        
        # Context features (previous tokens)
        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("formLowerPrev=" + tPrev.lower())
            tokenFeatures.append("casingPrev=" + casing(tPrev))
            
            # Bigram with previous word
            tokenFeatures.append("bigram=" + tPrev.lower() + "_" + t.lower())
            
            if k > 1:
                tPrev2 = tokens[k - 2][0]
                tokenFeatures.append("formPrev2=" + tPrev2)
                # Trigram with previous words
                tokenFeatures.append("trigram=" + tPrev2.lower() + "_" + tPrev.lower() + "_" + t.lower())
        else:
            tokenFeatures.append("BoS")

        # Context features (next tokens)
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

        # Surface window features
        surface1 = t.lower()
        surface2 = window_surface(tokens, k, 2)
        surface3 = window_surface(tokens, k, 3)
        back_window2 = backward_window(tokens, k, 2)
        
        # Enhanced lexicon features
        # Check exact matches
        if surface1 in drugbank_lexicon:
            tokenFeatures.append("exactInDrugBank=true")
            if surface1 in drugbank_types:
                tokenFeatures.append("drugType=" + drugbank_types[surface1])
        if surface1 in hsdb_lexicon:
            tokenFeatures.append("exactInHSDB=true")
            
        # Check window matches
        if surface2 in drugbank_lexicon or surface3 in drugbank_lexicon:
            tokenFeatures.append("windowInDrugBank=true")
        if surface2 in hsdb_lexicon or surface3 in hsdb_lexicon:
            tokenFeatures.append("windowInHSDB=true")
            
        # Check backward window matches
        if back_window2 in drugbank_lexicon:
            tokenFeatures.append("backWindowInDrugBank=true")
        if back_window2 in hsdb_lexicon:
            tokenFeatures.append("backWindowInHSDB=true")
            
        # Check for partial matches (substring) - using optimized functions
        if has_partial_match_db(surface1):
            tokenFeatures.append("partialMatchDrugBank=true")
        
        if has_partial_match_hsdb(surface1):
            tokenFeatures.append("partialMatchHSDB=true")

        result.append(tokenFeatures)

    return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
