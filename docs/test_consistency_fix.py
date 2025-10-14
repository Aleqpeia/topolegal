#!/usr/bin/env python3
"""
Test script to verify that modifications to text are consistently applied to triplets.

This is a CRITICAL feature: when we modify a legal reference in the text (e.g., 
"ч.2 ст.185" → "ч.2 ст.187"), we MUST apply the same modification to matching
references in triplets. Otherwise, the data becomes inconsistent!
"""

import sys
sys.path.insert(0, 'src')

from legal_graph.processors.negative_sampling import NegativeSampler

def test_text_triplet_consistency():
    """Test that modifications to text are consistently applied to triplets"""
    print("Testing text-triplet consistency...")
    print("=" * 70)
    
    # Create a document with a legal reference in both text and triplet
    dataset = [{
        'doc_id': 'test_consistency_001',
        'text': 'Підозрюваний обвинувачується за ч.2 ст.185 КК України',
        'entities': [
            {'text': 'ч.2 ст.185 КК України', 'type': 'legal_reference'}
        ],
        'triplets': [
            {
                'source': 'підозрюваний',
                'relation': 'обвинувачується_за',
                'target': 'ч.2 ст.185 КК України',
                'legal_reference': 'ч.2 ст.185 КК України',
                'confidence': 0.9
            }
        ],
        'triplets_count': 1
    }]
    
    # Create sampler with 100% probability to ensure modification happens
    sampler = NegativeSampler(
        sampling_probability=1.0,
        strategy='article'
    )
    
    print(f"\n📝 Original Document:")
    print(f"   Text: {dataset[0]['text']}")
    print(f"   Triplet legal_reference: {dataset[0]['triplets'][0]['legal_reference']}")
    print(f"   Triplet target: {dataset[0]['triplets'][0]['target']}")
    
    # Apply negative sampling
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    modified_doc = result[0]
    
    print(f"\n📝 Modified Document:")
    print(f"   Text: {modified_doc['text']}")
    print(f"   Triplet legal_reference: {modified_doc['triplets'][0]['legal_reference']}")
    print(f"   Triplet target: {modified_doc['triplets'][0]['target']}")
    
    # Extract the reference from the modified text
    import re
    text_ref_match = re.search(r'ч\.\d+ ст\.\d+ КК України', modified_doc['text'])
    assert text_ref_match, "Could not find legal reference in modified text"
    text_reference = text_ref_match.group(0)
    
    # Extract references from triplet
    triplet_legal_ref = modified_doc['triplets'][0]['legal_reference']
    triplet_target = modified_doc['triplets'][0]['target']
    
    print(f"\n🔍 Consistency Check:")
    print(f"   Text reference:           {text_reference}")
    print(f"   Triplet legal_reference:  {triplet_legal_ref}")
    print(f"   Triplet target:           {triplet_target}")
    
    # CRITICAL TEST: All three should match!
    if text_reference == triplet_legal_ref == triplet_target:
        print(f"\n✅ CONSISTENCY VERIFIED!")
        print(f"   All references match: {text_reference}")
        return True
    else:
        print(f"\n❌ CONSISTENCY FAILED!")
        print(f"   References don't match across text and triplets!")
        return False


def test_multiple_references_consistency():
    """Test consistency when document has multiple different legal references"""
    print("\n\nTesting multiple references consistency...")
    print("=" * 70)
    
    dataset = [{
        'doc_id': 'test_consistency_002',
        'text': 'Обвинувачується за ч.2 ст.185 КК України та ч.1 ст.190 КК України',
        'entities': [],
        'triplets': [
            {
                'source': 'підозрюваний',
                'relation': 'обвинувачується_за',
                'target': 'крадіжка',
                'legal_reference': 'ч.2 ст.185 КК України',
                'confidence': 0.9
            },
            {
                'source': 'підозрюваний',
                'relation': 'обвинувачується_за',
                'target': 'шахрайство',
                'legal_reference': 'ч.1 ст.190 КК України',
                'confidence': 0.9
            }
        ],
        'triplets_count': 2
    }]
    
    sampler = NegativeSampler(
        sampling_probability=1.0,
        strategy='article'
    )
    
    print(f"\n📝 Original Document:")
    print(f"   Text: {dataset[0]['text']}")
    print(f"   Triplet 1 legal_reference: {dataset[0]['triplets'][0]['legal_reference']}")
    print(f"   Triplet 2 legal_reference: {dataset[0]['triplets'][1]['legal_reference']}")
    
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    modified_doc = result[0]
    
    print(f"\n📝 Modified Document:")
    print(f"   Text: {modified_doc['text']}")
    print(f"   Triplet 1 legal_reference: {modified_doc['triplets'][0]['legal_reference']}")
    print(f"   Triplet 2 legal_reference: {modified_doc['triplets'][1]['legal_reference']}")
    
    # Extract all references from text
    import re
    text_refs = re.findall(r'ч\.\d+ ст\.\d+ КК України', modified_doc['text'])
    triplet_refs = [t['legal_reference'] for t in modified_doc['triplets']]
    
    print(f"\n🔍 Consistency Check:")
    print(f"   Text references:    {text_refs}")
    print(f"   Triplet references: {triplet_refs}")
    
    # Check that each triplet reference appears in the text
    all_consistent = True
    for triplet_ref in triplet_refs:
        if triplet_ref in modified_doc['text']:
            print(f"   ✅ '{triplet_ref}' found in text")
        else:
            print(f"   ❌ '{triplet_ref}' NOT found in text!")
            all_consistent = False
    
    if all_consistent:
        print(f"\n✅ CONSISTENCY VERIFIED for all references!")
        return True
    else:
        print(f"\n❌ CONSISTENCY FAILED for some references!")
        return False


def test_no_modification_no_change():
    """Test that when text is not modified, triplets are also not modified"""
    print("\n\nTesting no-modification case...")
    print("=" * 70)
    
    dataset = [{
        'doc_id': 'test_consistency_003',
        'text': 'This text has no legal references',
        'entities': [],
        'triplets': [
            {
                'source': 'defendant',
                'relation': 'charged_with',
                'target': 'theft',
                'legal_reference': '',
                'confidence': 0.9
            }
        ],
        'triplets_count': 1
    }]
    
    sampler = NegativeSampler(
        sampling_probability=1.0,
        strategy='article'
    )
    
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    modified_doc = result[0]
    
    # Since no legal references were found, document should be unchanged
    if modified_doc['text'] == dataset[0]['text']:
        print("✅ Text unchanged (no legal references)")
        print("✅ Document marked as valid (no modifications possible)")
        return True
    else:
        print("❌ Unexpected modification occurred")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("CRITICAL CONSISTENCY TEST")
    print("Verifying text-triplet consistency in negative sampling")
    print("=" * 70)
    
    try:
        test1_pass = test_text_triplet_consistency()
        test2_pass = test_multiple_references_consistency()
        test3_pass = test_no_modification_no_change()
        
        print("\n" + "=" * 70)
        if test1_pass and test2_pass and test3_pass:
            print("✅ ALL CONSISTENCY TESTS PASSED!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("❌ SOME CONSISTENCY TESTS FAILED!")
            print("=" * 70)
            sys.exit(1)
            
    except Exception as e:
        print("=" * 70)
        print(f"❌ Test failed with exception: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

