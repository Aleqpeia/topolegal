import pytest
import spacy
from processors.entities.person import PersonComponent

def test_person_component():
    """Test that the PersonComponent correctly identifies person entities."""
    # Create a blank spaCy model
    nlp = spacy.blank("uk")
    
    # Add the person component to the pipeline
    nlp.add_pipe("person_component")
    
    # Test text with a person entity
    text = "У справі ОСОБА_1 проти ОСОБА_2"
    doc = nlp(text)
    
    # Check that entities were found
    assert len(doc.ents) == 2
    
    # Check that the entities have the correct label
    assert all(ent.label_ == "PER" for ent in doc.ents)
    
    # Check that the entities have the correct text
    assert doc.ents[0].text == "ОСОБА_1"
    assert doc.ents[1].text == "ОСОБА_2"
    
    # Test text without person entities
    text_without_entities = "У цій справі немає осіб."
    doc_without_entities = nlp(text_without_entities)
    
    # Check that no entities were found
    assert len(doc_without_entities.ents) == 0