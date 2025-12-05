import re
import spacy

# Load the NLP model once (Global variable)
# "en_core_web_sm" is a small English model trained on web text
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ SpaCy model not found. Downloading it now...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_information(text, category):
    """
    Extracts key fields using a Hybrid approach:
    1. SpaCy NER (Named Entity Recognition) -> For People & Companies.
    2. Regex (Pattern Matching) -> For Dates, Emails, Money.
    """
    results = {}
    
    # --- 1. SPACY (AI NER) ---
    doc = nlp(text)
    
    # Extract distinct entities to avoid duplicates
    people = list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"]))
    orgs = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))
    
    # Filter out noise (short words that aren't names)
    people = [p for p in people if len(p) > 2]
    orgs = [o for o in orgs if len(o) > 2]

    # Add to results if found
    if people:
        results["People/Names"] = people[:5] # Limit to top 5
    if orgs:
        results["Organizations"] = orgs[:5]

    # --- 2. REGEX (PATTERNS) ---
    text_lines = text.split('\n')
    
    # EMAIL Category Specifics
    if category == "email" or category == "resume":
        emails = re.findall(r'[a-zA-Z0-9._%+-]+[\s]?[@|©]?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        clean_emails = [e.replace(" ", "").replace("©", "@") for e in emails if "@" in e.replace("©", "@")]
        if clean_emails:
            results["Emails"] = list(set(clean_emails))

        # Find "Subject" line (Only for emails)
        if category == "email":
            for line in text_lines:
                if "Subject:" in line or "Re:" in line:
                    results["Subject"] = line.strip()
                    break

    # INVOICE Category Specifics
    if category == "invoice":
        # Find Money ($500.00)
        amounts = re.findall(r'\$\s?([0-9,]+\.[0-9]{2})', text)
        if amounts:
            try:
                floats = [float(a.replace(',', '')) for a in amounts]
                results["Total Amount"] = f"${max(floats):,.2f}"
            except:
                pass
        
        # Find Dates (mm/dd/yyyy)
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        if dates:
            results["Dates"] = dates[:3]

    # RESUME Category Specifics
    if category == "resume":
        phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phones:
            results["Phone"] = phones[0]

    return results