#!/usr/bin/env python3
"""
Quick test to verify text is being modified
"""

import sys
sys.path.insert(0, 'src')

from legal_graph.processors.negative_sampling import NegativeSampler

# Test case 1: Basic modification
dataset = [{
    'doc_id': 'test_001',
    'text': 'Підозрюваний обвинувачується за ч.2 ст.185 КК України',
    'triplets': [
        {
            'legal_reference': 'ч.2 ст.185 КК України',
            'relation': 'обвинувачується_за'
        }
    ],
    'triplets_count': 1
}]

sampler = NegativeSampler(sampling_probability=1.0, strategy='article')

print("Original text:", dataset[0]['text'])
print("Original triplet:", dataset[0]['triplets'][0]['legal_reference'])

result = sampler.create_negative_samples(dataset, label_column='is_valid')

print("\nModified text:", result[0]['text'])
print("Modified triplet:", result[0]['triplets'][0]['legal_reference'])
print("Is valid:", result[0]['is_valid'])

# Check if text was actually modified
if result[0]['text'] != dataset[0]['text']:
    print("\n✅ TEXT WAS MODIFIED")
else:
    print("\n❌ TEXT WAS NOT MODIFIED!")

# Check if triplet was modified
if result[0]['triplets'][0]['legal_reference'] != dataset[0]['triplets'][0]['legal_reference']:
    print("✅ TRIPLET WAS MODIFIED")
else:
    print("❌ TRIPLET WAS NOT MODIFIED!")

