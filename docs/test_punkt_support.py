#!/usr/bin/env python3
"""
Test script to verify support for "п." (punkt) format in addition to "ст." (stattya)

Tests:
1. "ч.X ст.Y КК України" - part + article (already supported)
2. "ч.X п.Y КК України" - part + punkt (NEW)
3. "п.Y КК України" - standalone punkt (NEW)
4. Text-triplet consistency for all formats
"""

import sys
sys.path.insert(0, 'src')

from legal_graph.processors.negative_sampling import NegativeSampler

def test_part_article_format():
    """Test traditional 'ч.X ст.Y' format"""
    print("=" * 70)
    print("Test 1: Traditional 'ч.X ст.Y КК України' format")
    print("=" * 70)
    
    dataset = [{
        'doc_id': 'test_001',
        'text': 'Обвинувачується за ч.2 ст.185 КК України',
        'triplets': [{'legal_reference': 'ч.2 ст.185 КК України'}],
        'triplets_count': 1
    }]
    
    sampler = NegativeSampler(sampling_probability=1.0, strategy='article')
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    
    print(f"Original text: {dataset[0]['text']}")
    print(f"Modified text: {result[0]['text']}")
    print(f"Original triplet: {dataset[0]['triplets'][0]['legal_reference']}")
    print(f"Modified triplet: {result[0]['triplets'][0]['legal_reference']}")
    
    # Extract references
    import re
    text_ref = re.search(r'ч\.\d+ ст\.\d+ КК України', result[0]['text']).group(0)
    triplet_ref = result[0]['triplets'][0]['legal_reference']
    
    if text_ref == triplet_ref and text_ref != dataset[0]['triplets'][0]['legal_reference']:
        print("✅ PASS: Text and triplet match, and both were modified")
        return True
    else:
        print("❌ FAIL: Inconsistency or no modification")
        return False


def test_part_punkt_format():
    """Test 'ч.X п.Y' format (punkt instead of stattya)"""
    print("\n" + "=" * 70)
    print("Test 2: 'ч.X п.Y КК України' format (punkt)")
    print("=" * 70)
    
    dataset = [{
        'doc_id': 'test_002',
        'text': 'Згідно з ч.1 п.5 КК України підозрюваний притягується',
        'triplets': [{'legal_reference': 'ч.1 п.5 КК України'}],
        'triplets_count': 1
    }]
    
    sampler = NegativeSampler(sampling_probability=1.0, strategy='article')
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    
    print(f"Original text: {dataset[0]['text']}")
    print(f"Modified text: {result[0]['text']}")
    print(f"Original triplet: {dataset[0]['triplets'][0]['legal_reference']}")
    print(f"Modified triplet: {result[0]['triplets'][0]['legal_reference']}")
    
    # Extract references
    import re
    text_match = re.search(r'ч\.\d+ п\.\d+ КК України', result[0]['text'])
    
    if text_match:
        text_ref = text_match.group(0)
        triplet_ref = result[0]['triplets'][0]['legal_reference']
        
        if text_ref == triplet_ref and text_ref != dataset[0]['triplets'][0]['legal_reference']:
            print("✅ PASS: Text and triplet match, and both were modified")
            return True
        else:
            print(f"❌ FAIL: Text ref: '{text_ref}', Triplet ref: '{triplet_ref}'")
            return False
    else:
        print("❌ FAIL: Could not find 'ч.X п.Y' pattern in modified text")
        return False


def test_standalone_punkt_format():
    """Test standalone 'п.Y' format"""
    print("\n" + "=" * 70)
    print("Test 3: Standalone 'п.Y КК України' format")
    print("=" * 70)
    
    dataset = [{
        'doc_id': 'test_003',
        'text': 'Відповідно до п.12 КК України застосовується міра покарання',
        'triplets': [{'legal_reference': 'п.12 КК України'}],
        'triplets_count': 1
    }]
    
    sampler = NegativeSampler(sampling_probability=1.0, strategy='article')
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    
    print(f"Original text: {dataset[0]['text']}")
    print(f"Modified text: {result[0]['text']}")
    print(f"Original triplet: {dataset[0]['triplets'][0]['legal_reference']}")
    print(f"Modified triplet: {result[0]['triplets'][0]['legal_reference']}")
    
    # Extract references
    import re
    text_match = re.search(r'п\.\d+ КК України', result[0]['text'])
    
    if text_match:
        text_ref = text_match.group(0)
        triplet_ref = result[0]['triplets'][0]['legal_reference']
        
        if text_ref == triplet_ref and text_ref != dataset[0]['triplets'][0]['legal_reference']:
            print("✅ PASS: Text and triplet match, and both were modified")
            return True
        else:
            print(f"❌ FAIL: Text ref: '{text_ref}', Triplet ref: '{triplet_ref}'")
            return False
    else:
        print("❌ FAIL: Could not find 'п.Y' pattern in modified text")
        return False


def test_mixed_formats():
    """Test document with both formats"""
    print("\n" + "=" * 70)
    print("Test 4: Mixed formats in one document")
    print("=" * 70)
    
    dataset = [{
        'doc_id': 'test_004',
        'text': 'За ч.2 ст.185 КК України та п.12 КК України',
        'triplets': [
            {'legal_reference': 'ч.2 ст.185 КК України'},
            {'legal_reference': 'п.12 КК України'}
        ],
        'triplets_count': 2
    }]
    
    sampler = NegativeSampler(sampling_probability=1.0, strategy='article')
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    
    print(f"Original text: {dataset[0]['text']}")
    print(f"Modified text: {result[0]['text']}")
    print(f"Original triplets: {[t['legal_reference'] for t in dataset[0]['triplets']]}")
    print(f"Modified triplets: {[t['legal_reference'] for t in result[0]['triplets']]}")
    
    # Check that all triplet references appear in text
    all_consistent = True
    for triplet in result[0]['triplets']:
        ref = triplet['legal_reference']
        if ref in result[0]['text']:
            print(f"  ✅ '{ref}' found in text")
        else:
            print(f"  ❌ '{ref}' NOT found in text!")
            all_consistent = False
    
    if all_consistent:
        print("✅ PASS: All references consistent")
        return True
    else:
        print("❌ FAIL: Some references inconsistent")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PUNKT FORMAT SUPPORT TEST")
    print("Testing 'п.' (punkt) in addition to 'ст.' (stattya)")
    print("=" * 70)
    
    try:
        test1 = test_part_article_format()
        test2 = test_part_punkt_format()
        test3 = test_standalone_punkt_format()
        test4 = test_mixed_formats()
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Test 1 (ч.X ст.Y): {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"Test 2 (ч.X п.Y):  {'✅ PASS' if test2 else '❌ FAIL'}")
        print(f"Test 3 (п.Y):      {'✅ PASS' if test3 else '❌ FAIL'}")
        print(f"Test 4 (Mixed):    {'✅ PASS' if test4 else '❌ FAIL'}")
        
        if test1 and test2 and test3 and test4:
            print("\n✅ ALL TESTS PASSED!")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

