"""
Entry point for legal-text NER extraction using custom spaCy components.

Usage:
    python -m processors.entities [INPUT_FILE]
    cat file.txt | python -m processors.entities
"""
import sys
import spacy

# Register all custom components (module-level factories)
import processors.entities.person
import processors.entities.date
import processors.entities.case
import processors.entities.role
import processors.entities.doctype
import processors.entities.crime


def build_pipeline():
    # You can switch to a pretrained model:
    # nlp = spacy.load("uk_core_news_sm")
    nlp = spacy.blank("uk")
    # Add components in desired order
    for name in [
        "person_component",
        "date_component",
        "case_component",
        "role_component",
        "doctype_component",
        "crime_component",
    ]:
        nlp.add_pipe(name, last=True)
    return nlp


def main():
    # Read from file or stdin
    if len(sys.argv) > 1:
        text = open(sys.argv[1], encoding="utf-8").read()
    else:
        text = sys.stdin.read()

    nlp = build_pipeline()
    doc = nlp(text)

    # Simple TSV output: label, text, start, end
    for ent in doc.ents:
        print(f"{ent.label_}\t{ent.text}\t{ent.start_char}\t{ent.end_char}")


if __name__ == "__main__":
    main()
