# Chembl
SELECT 
    md.MOLREGNO,
    md.PREF_NAME,
    COALESCE(group_concat(ms.SYNONYMS, '; '), '') AS Synonyms,
    md.MAX_PHASE AS Approval_Phase
FROM 
    MOLECULE_DICTIONARY md
    LEFT JOIN MOLECULE_SYNONYMS ms 
        ON md.MOLREGNO = ms.MOLREGNO
    LEFT JOIN DRUG_INDICATION di 
        ON md.MOLREGNO = di.MOLREGNO
    LEFT JOIN MOLECULE_HIERARCHY mh 
        ON md.MOLREGNO = mh.MOLREGNO
WHERE 
    -- Select only compounds with approved status
    md.MAX_PHASE = 4
    -- Ensure any indication record also shows phase 4 (or is absent)
    AND (di.MAX_PHASE_FOR_IND IS NULL OR di.MAX_PHASE_FOR_IND = 4)
    -- Limit to primary compounds (i.e. where no parent is defined or compound equals its parent)
    AND (mh.PARENT_MOLREGNO IS NULL OR md.MOLREGNO = mh.PARENT_MOLREGNO)
GROUP BY 
    md.MOLREGNO,
    md.PREF_NAME,
    md.MAX_PHASE
ORDER BY 
    md.PREF_NAME;

# Drugbank
1. **Parsing DrugBank XML:**
   - The script parses the DrugBank XML file (`drugbank.xml`) using Python's `xml.etree.ElementTree` with the appropriate XML namespace.

2. **Filtering Approved Drugs:**
   - For each drug entry, it checks if the drug is marked as "approved" in the `<groups>` section.

3. **Extracting Names and Synonyms:**
   - If a drug is approved, the script extracts:
     - The main drug name
     - All synonyms listed under `<synonyms>`
     - All product names under `<products>`

4. **Deduplication:**
   - All names are stored in a set to ensure uniqueness (case-insensitive sort for output).

5. **Output:**
   - The unique names are written to `approved_drugs_drugbank.txt`, one per line, in UTF-8 encoding.
