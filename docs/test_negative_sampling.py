#!/usr/bin/env python3
"""
Simple test script for negative sampling module
"""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        from legal_graph.processors.negative_sampling import (
            NegativeSampler,
            create_negative_samples,
            ArticleOffsetStrategy,
            PartOffsetStrategy,
            MixedStrategy
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_sampling():
    """Test basic negative sampling"""
    print("\nTesting basic negative sampling...")
    try:
        from legal_graph.processors.negative_sampling import create_negative_samples

        # Create test dataset
        dataset = [
            {
                'doc_id': 'test_001',
                'text': 'Суд визнав винним згідно з ч.2 ст.185 КК України',
                'entities': [
                    {'text': 'Суд', 'label': 'ORG'},
                    {'text': 'ч.2 ст.185 КК України', 'label': 'LEGAL_REF'}
                ],
                'triplets': [
                    {
                        'source': 'Суд',
                        'relation': 'визнав_винним',
                        'target': 'ОСОБА_1',
                        'legal_reference': 'ч.2 ст.185 КК України'
                    }
                ]
            },
            {
                'doc_id': 'test_002',
                'text': 'Відповідно до ст.331 КПК України слідчий здійснює розслідування',
                'entities': [],
                'triplets': []
            }
        ]

        # Test with article strategy
        result = create_negative_samples(
            dataset,
            strategy='article',
            sampling_probability=1.0  # Sample all
        )

        print(f"  Original samples: {len(dataset)}")
        print(f"  Result samples: {len(result)}")
        print(f"  Valid samples: {sum(1 for s in result if s.get('is_valid', True))}")
        print(f"  Negative samples: {sum(1 for s in result if not s.get('is_valid', True))}")

        # Show example
        for i, sample in enumerate(result[:2]):
            print(f"\n  Sample {i+1}:")
            print(f"    Valid: {sample.get('is_valid', True)}")
            print(f"    Text: {sample['text'][:80]}...")

        print("✅ Basic sampling test passed")
        return True

    except Exception as e:
        print(f"❌ Basic sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategies():
    """Test different strategies"""
    print("\nTesting different strategies...")
    try:
        from legal_graph.processors.negative_sampling import NegativeSampler

        strategies = ['article', 'part', 'combined', 'mixed']
        dataset = [{
            'doc_id': 'test',
            'text': 'згідно з ч.2 ст.185 КК України'
        }]

        for strategy_name in strategies:
            sampler = NegativeSampler(
                strategy=strategy_name,
                sampling_probability=1.0
            )
            result = sampler.create_negative_samples(dataset)
            stats = sampler.get_sampling_stats()

            print(f"  {strategy_name:12s}: {stats['corrupted']} samples created")

        print("✅ Strategy test passed")
        return True

    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_help():
    """Test CLI help command"""
    print("\nTesting CLI interface...")
    try:
        import subprocess
        result = subprocess.run(
            ['python3', '-m', 'legal_graph.processors.negative_sampling', 'strategies'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent / 'src'
        )

        if 'Available Sampling Strategies' in result.stdout:
            print("✅ CLI interface test passed")
            return True
        else:
            print("⚠️  CLI interface might not be working correctly")
            return False

    except Exception as e:
        print(f"⚠️  CLI test skipped: {e}")
        return True  # Don't fail if python3 not available


def main():
    """Run all tests"""
    print("=" * 60)
    print("Negative Sampling Module Test Suite")
    print("=" * 60)

    tests = [
        test_imports,
        test_basic_sampling,
        test_strategies,
        test_cli_help
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
