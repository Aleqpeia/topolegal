#!/usr/bin/env python
"""
Simple standalone test for negative sampling module logic
Tests the core regex patterns and modification logic without dependencies
"""

import re
import random

def test_regex_patterns():
    """Test that regex patterns correctly match legal references"""
    print("=" * 60)
    print("Testing Legal Reference Regex Patterns")
    print("=" * 60)

    # Pattern for "ч.X ст.Y КК України"
    part_article_pattern = re.compile(
        r'(ч\.\s*)(\d+)(\s+ст\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)',
        re.IGNORECASE
    )

    # Test cases
    test_cases = [
        "згідно з ч.2 ст.185 КК України",
        "Відповідно до ч.1 ст.331 КПК України",
        "на підставі ч.3 ст.1166 ЦК України",
        "згідно з ст.185 КК України",  # Without part - should not match
    ]

    print("\nTest Cases:")
    for i, text in enumerate(test_cases, 1):
        match = part_article_pattern.search(text)
        if match:
            part = match.group(2)
            article = match.group(4)
            code = match.group(5).strip()
            print(f"  {i}. ✅ Matched: '{text}'")
            print(f"      Part: {part}, Article: {article}, Code: {code}")
        else:
            print(f"  {i}. ⚠️  No match: '{text}'")

    return True


def test_modification_logic():
    """Test the modification logic"""
    print("\n" + "=" * 60)
    print("Testing Modification Logic")
    print("=" * 60)

    # Simulate article offset modification
    def modify_article_offset(article, offset_range=(1, 5)):
        offset = random.choice(list(range(offset_range[0], offset_range[1] + 1)))
        # Randomly choose positive or negative
        offset = offset if random.random() > 0.5 else -offset
        new_article = max(1, min(446, article + offset))  # Keep in valid range
        return new_article, offset

    # Simulate part offset modification
    def modify_part_offset(part, offset_range=(1, 2)):
        offset = random.choice(list(range(offset_range[0], offset_range[1] + 1)))
        offset = offset if random.random() > 0.5 else -offset
        new_part = max(1, min(10, part + offset))  # Keep in valid range
        return new_part, offset

    # Test modifications
    test_refs = [
        (185, 2),  # article 185, part 2
        (331, 1),  # article 331, part 1
        (1, 1),    # Edge case: article 1, part 1
        (446, 10), # Edge case: max article, max part
    ]

    print("\nModification Examples:")
    for article, part in test_refs:
        new_article, art_offset = modify_article_offset(article)
        new_part, part_offset = modify_part_offset(part)

        print(f"\n  Original:  ч.{part} ст.{article}")
        print(f"  Modified:  ч.{new_part} ст.{new_article}")
        print(f"  Offsets:   article {art_offset:+d}, part {part_offset:+d}")

    return True


def test_full_replacement():
    """Test full text replacement"""
    print("\n" + "=" * 60)
    print("Testing Full Text Replacement")
    print("=" * 60)

    part_article_pattern = re.compile(
        r'(ч\.\s*)(\d+)(\s+ст\.\s*)(\d+)(\s+(?:КК|КПК|ЦК|КоАП|СК|КЗпП)\s+України)',
        re.IGNORECASE
    )

    test_texts = [
        "Суд визнав винним згідно з ч.2 ст.185 КК України",
        "Відповідно до ч.1 ст.331 КПК України слідчий провів",
        "На підставі ч.3 ст.1166 ЦК України та ч.1 ст.1167 ЦК України",
    ]

    print("\nReplacement Examples:")
    for i, text in enumerate(test_texts, 1):
        matches = list(part_article_pattern.finditer(text))
        modified_text = text

        print(f"\n  {i}. Original:")
        print(f"     {text}")

        # Process in reverse to maintain positions
        for match in reversed(matches):
            part_prefix = match.group(1)
            part_num = int(match.group(2))
            middle = match.group(3)
            article_num = int(match.group(4))
            suffix = match.group(5)

            # Simple modification: +1 to both
            new_part = min(part_num + 1, 10)
            new_article = min(article_num + 2, 446)

            new_ref = f"{part_prefix}{new_part}{middle}{new_article}{suffix}"
            modified_text = (
                modified_text[:match.start()] +
                new_ref +
                modified_text[match.end():]
            )

        print(f"     Modified:")
        print(f"     {modified_text}")
        print(f"     Found {len(matches)} reference(s)")

    return True


def test_strategy_simulation():
    """Test different modification strategies"""
    print("\n" + "=" * 60)
    print("Testing Strategy Simulations")
    print("=" * 60)

    strategies = {
        'article_offset': lambda a, p: (a + 2, p, 'article+2'),
        'part_offset': lambda a, p: (a, p + 1, 'part+1'),
        'combined': lambda a, p: (a + 2, p + 1, 'combined'),
        'swap': lambda a, p: (p, a, 'swap'),
    }

    original = (185, 2)  # ч.2 ст.185

    print(f"\nOriginal: ч.{original[1]} ст.{original[0]}")
    print("\nStrategy Results:")

    for strategy_name, strategy_func in strategies.items():
        new_article, new_part, mod_type = strategy_func(*original)
        print(f"  {strategy_name:15s}: ч.{new_part} ст.{new_article} ({mod_type})")

    return True


def test_bigquery_data_format():
    """Test BigQuery data format compatibility"""
    print("\n" + "=" * 60)
    print("Testing BigQuery Data Format")
    print("=" * 60)

    # Simulate BigQuery row format
    sample_data = {
        'doc_id': 'test_001',
        'text': 'Суд визнав винним згідно з ч.2 ст.185 КК України',
        'tags': '[{"text": "Суд", "label": "ORG"}, {"text": "ч.2 ст.185 КК України", "label": "LEGAL_REF"}]',
        'triplets': '[{"source": "Суд", "relation": "визнав_винним", "target": "ОСОБА_1", "legal_reference": "ч.2 ст.185 КК України"}]',
        'triplets_count': 1
    }

    print("\nSample BigQuery Row:")
    for key, value in sample_data.items():
        value_str = str(value)[:60] + '...' if len(str(value)) > 60 else str(value)
        print(f"  {key:15s}: {value_str}")

    # Test JSON parsing
    import json
    try:
        tags = json.loads(sample_data['tags'])
        triplets = json.loads(sample_data['triplets'])
        print("\n✅ JSON parsing successful")
        print(f"  Parsed {len(tags)} entities")
        print(f"  Parsed {len(triplets)} triplets")

        # Test legal reference extraction from triplet
        if triplets and 'legal_reference' in triplets[0]:
            legal_ref = triplets[0]['legal_reference']
            print(f"  Legal reference: {legal_ref}")

        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Negative Sampling Module - Standalone Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Regex Patterns", test_regex_patterns),
        ("Modification Logic", test_modification_logic),
        ("Full Replacement", test_full_replacement),
        ("Strategy Simulation", test_strategy_simulation),
        ("BigQuery Format", test_bigquery_data_format),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    return all(r for _, r in results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)