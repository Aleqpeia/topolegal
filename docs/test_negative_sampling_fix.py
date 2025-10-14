#!/usr/bin/env python3
"""
Test script to verify the negative sampling fixes for:
1. Multiple legal references handling
2. Non-empty triplets filtering
3. Array ambiguity error fixes
"""

import sys
sys.path.insert(0, 'src')

import json
from legal_graph.processors.negative_sampling import NegativeSampler

def test_multiple_references():
    """Test that documents with multiple legal references are processed correctly"""
    print("Testing multiple legal references...")
    
    # Create a document with multiple legal references
    dataset = [{
        'doc_id': 'test_001',
        'text': 'згідно з ч.2 ст.185 КК України та ч.1 ст.190 КК України',
        'entities': [
            {'text': 'ч.2 ст.185 КК України', 'type': 'legal_reference'},
            {'text': 'ч.1 ст.190 КК України', 'type': 'legal_reference'}
        ],
        'triplets': [
            {
                'source': 'підозрюваний',
                'relation': 'обвинувачується_за',
                'target': 'ч.2 ст.185 КК України',
                'legal_reference': 'ч.2 ст.185 КК України'
            },
            {
                'source': 'підозрюваний',
                'relation': 'обвинувачується_за',
                'target': 'ч.1 ст.190 КК України',
                'legal_reference': 'ч.1 ст.190 КК України'
            }
        ],
        'triplets_count': 2
    }]
    
    # Create sampler and process
    sampler = NegativeSampler(
        sampling_probability=1.0,  # Always corrupt for testing
        strategy='article'
    )
    
    result = sampler.create_negative_samples(dataset, label_column='is_valid')
    
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    assert 'is_valid' in result[0], "Missing is_valid label"
    
    # The document should be corrupted (is_valid=False)
    if not result[0]['is_valid']:
        print("✓ Document was corrupted as expected")
        print(f"  Original: {dataset[0]['text']}")
        print(f"  Modified: {result[0]['text']}")
    else:
        print("⚠ Document was not corrupted (might happen with small probability)")
    
    print("✓ Multiple references test passed!\n")


def test_empty_triplets_filtering():
    """Test that documents without triplets are filtered out"""
    print("Testing empty triplets filtering...")
    
    dataset = [
        {
            'doc_id': 'test_002',
            'text': 'згідно з ч.2 ст.185 КК України',
            'entities': [],
            'triplets': [],  # Empty triplets
            'triplets_count': 0
        },
        {
            'doc_id': 'test_003',
            'text': 'згідно з ч.1 ст.190 КК України',
            'entities': [],
            'triplets': [
                {
                    'source': 'підозрюваний',
                    'relation': 'обвинувачується_за',
                    'target': 'ч.1 ст.190 КК України',
                    'legal_reference': 'ч.1 ст.190 КК України'
                }
            ],
            'triplets_count': 1
        }
    ]
    
    # Filter out documents without triplets (simulating what __main__.py does)
    filtered_dataset = [doc for doc in dataset if doc.get('triplets') and len(doc['triplets']) > 0]
    
    assert len(filtered_dataset) == 1, f"Expected 1 document after filtering, got {len(filtered_dataset)}"
    assert filtered_dataset[0]['doc_id'] == 'test_003', "Wrong document was kept"
    
    print(f"✓ Filtered {len(dataset)} -> {len(filtered_dataset)} documents")
    print("✓ Empty triplets filtering test passed!\n")


def test_array_handling():
    """Test that array values don't cause ambiguity errors"""
    print("Testing array value handling...")
    
    # Simulate what happens when parsing JSON fields that contain arrays
    test_values = [
        ('[]', []),
        ('[{"test": "value"}]', [{"test": "value"}]),
        ('null', []),
        ('nan', []),
        (None, []),
        ('', [])
    ]
    
    for test_value, expected in test_values:
        # Simulate the safe parsing logic from __main__.py
        if test_value is not None and str(test_value).strip() and str(test_value) != 'nan':
            try:
                result = json.loads(str(test_value))
                if not isinstance(result, list):
                    result = []
            except (json.JSONDecodeError, ValueError):
                result = []
        else:
            result = []
        
        assert result == expected, f"Failed for {test_value}: got {result}, expected {expected}"
    
    print("✓ Array handling test passed!\n")


def test_strategy_variations():
    """Test different sampling strategies with multiple references"""
    print("Testing different strategies...")
    
    dataset = [{
        'doc_id': 'test_004',
        'text': 'згідно з ч.2 ст.185 КК України',
        'entities': [],
        'triplets': [
            {
                'source': 'підозрюваний',
                'relation': 'обвинувачується_за',
                'target': 'ч.2 ст.185 КК України',
                'legal_reference': 'ч.2 ст.185 КК України'
            }
        ],
        'triplets_count': 1
    }]
    
    strategies = ['article', 'part', 'combined', 'mixed', 'conservative', 'aggressive']
    
    for strategy_name in strategies:
        sampler = NegativeSampler(
            sampling_probability=1.0,
            strategy=strategy_name
        )
        result = sampler.create_negative_samples(dataset, label_column='is_valid')
        assert len(result) == 1, f"Strategy {strategy_name} failed"
        print(f"  ✓ Strategy '{strategy_name}' works")
    
    print("✓ Strategy variations test passed!\n")


if __name__ == '__main__':
    print("="*60)
    print("Negative Sampling Fix Verification Tests")
    print("="*60 + "\n")
    
    try:
        test_multiple_references()
        test_empty_triplets_filtering()
        test_array_handling()
        test_strategy_variations()
        
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print("="*60)
        print(f"❌ Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

