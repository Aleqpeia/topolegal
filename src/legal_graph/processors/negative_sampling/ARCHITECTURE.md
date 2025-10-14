# Negative Sampling Module Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TopoLegal Processors Module                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │   Entities     │  │ Graph Extract  │  │  Balance (Corrupt) │   │
│  │   (NER/spaCy)  │  │  (LLM/Triplets)│  │ (Random article)   │   │
│  └────────────────┘  └────────────────┘  └────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        Negative Sampling (Hard Negatives) - NEW             │   │
│  │                                                               │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │              NegativeSampler (Core)                   │  │   │
│  │  │  • Extends DataCorruptor ABC                          │  │   │
│  │  │  • Component-level modifications                      │  │   │
│  │  │  • Strategy pattern implementation                    │  │   │
│  │  │  • Statistics tracking                                │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                           │                                   │   │
│  │                           ▼                                   │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │           Sampling Strategies (8 types)               │  │   │
│  │  │                                                        │  │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │   │
│  │  │  │   Article    │  │     Part     │  │ Combined  │  │  │   │
│  │  │  │  Offset      │  │   Offset     │  │  Offset   │  │  │   │
│  │  │  │  (±1-5)      │  │   (±1-2)     │  │  (both)   │  │  │   │
│  │  │  └──────────────┘  └──────────────┘  └───────────┘  │  │   │
│  │  │                                                        │  │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │   │
│  │  │  │  Index Swap  │  │    Random    │  │   Mixed   │  │  │   │
│  │  │  │  (invalid)   │  │  Component   │  │ (Default) │  │  │   │
│  │  │  └──────────────┘  └──────────────┘  └───────────┘  │  │   │
│  │  │                                                        │  │   │
│  │  │  ┌──────────────┐  ┌──────────────┐                  │  │   │
│  │  │  │Conservative  │  │ Aggressive   │                  │  │   │
│  │  │  │   (subtle)   │  │  (maximum)   │                  │  │   │
│  │  │  └──────────────┘  └──────────────┘                  │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                               │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │              CLI Interface (__main__.py)              │  │   │
│  │  │  • sample-csv      • sample-json                      │  │   │
│  │  │  • sample-bigquery • strategies                       │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           DocumentProcessor (Unified Interface)             │   │
│  │  • extract_entities()                                       │   │
│  │  • extract_knowledge_graph()                                │   │
│  │  • corrupt_data()           [Balance module]                │   │
│  │  • create_negative_samples() [Negative sampling - NEW]     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Document
    │
    ├─ Text: "згідно з ч.2 ст.185 КК України"
    ├─ Entities: [...]
    └─ Triplets: [...]
    │
    ▼
┌───────────────────────────────────────┐
│      NegativeSampler.corrupt_sample() │
└───────────────────────────────────────┘
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
_modify_text()                     _modify_triplets()
    │                                     │
    │  1. Detect patterns                 │  Parse JSON/list
    │     ч.X ст.Y                        │  Iterate triplets
    │                                     │  Modify legal_reference field
    │  2. Extract components              │
    │     part=2, article=185             │
    │                                     │
    │  3. Apply strategy                  │
    │     strategy.modify_reference()     │
    │     → new_part=3, new_article=187   │
    │                                     │
    │  4. Reconstruct reference           │
    │     "ч.3 ст.187 КК України"         │
    │                                     │
    │  5. Replace in text                 │
    │                                     │
    ├─────────────────────────────────────┤
    │                                     │
    ▼                                     ▼
Modifications: [                  Modifications: [
  {                                 {
    'field': 'text',                  'field': 'triplet_legal_reference',
    'original': 'ч.2 ст.185...',      'original': 'ч.2 ст.185...',
    'modified': 'ч.3 ст.187...',      'modified': 'ч.3 ст.187...',
    'modification_type': 'combined'   'modification_type': 'combined'
  }                                 }
]                                 ]
    │                                     │
    └─────────────────┬───────────────────┘
                      ▼
            CorruptionResult
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
original_data   corrupted_data   corruption_details
    │                 │                 │
    └─────────────────┴─────────────────┘
                      ▼
            Output Dataset
            [
              {
                'doc_id': '001',
                'text': 'згідно з ч.3 ст.187 КК України',
                'is_valid': False,  # Negative sample
                ...
              },
              {
                'doc_id': '002',
                'text': 'згідно з ч.2 ст.185 КК України',
                'is_valid': True,   # Original
                ...
              }
            ]
```

## Strategy Selection Flow

```
                    NegativeSampler.__init__()
                              │
                    ┌─────────┴─────────┐
                    │   strategy param  │
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    String name         SamplingStrategy      Pre-defined
    'article'              instance          DEFAULT_STRATEGY
    'part'             (custom class)        CONSERVATIVE_STRATEGY
    'mixed'                                  AGGRESSIVE_STRATEGY
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                     create_strategy()
                              │
                    ┌─────────┴─────────┐
                    │  Strategy Factory │
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┬───────────────┐
          │                   │                   │               │
          ▼                   ▼                   ▼               ▼
    ArticleOffset      PartOffset         CombinedOffset   IndexSwap
          │                   │                   │               │
          └───────────────────┼───────────────────┴───────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Strategy Instance│
                    └─────────┬─────────┘
                              │
                              ▼
              Used in corrupt_sample() → _modify_text()
                              │
                    ┌─────────┴─────────┐
                    │ modify_reference()│
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┬─────────────────┐
                    │                   │                 │
                    ▼                   ▼                 ▼
              new_article          new_part        modification_type
              (int)                (int/None)      (string)
```

## Class Hierarchy

```
abc.ABC
    │
    ├─ DataCorruptor (from balance module)
    │       │
    │       ├─ can_corrupt() [abstract]
    │       ├─ corrupt_sample() [abstract]
    │       ├─ corrupt_dataset()
    │       └─ get_corruption_stats()
    │               │
    │               ├─ LegalReferenceCorruptor (balance)
    │               │
    │               └─ NegativeSampler (NEW)
    │                       │
    │                       ├─ _modify_text()
    │                       ├─ _modify_triplets()
    │                       ├─ _modify_entities()
    │                       ├─ create_negative_samples()
    │                       └─ get_sampling_stats()
    │
    └─ SamplingStrategy (NEW)
            │
            ├─ modify_reference() [abstract]
            └─ get_strategy_name() [abstract]
                    │
                    ├─ ArticleOffsetStrategy
                    ├─ PartOffsetStrategy
                    ├─ CombinedOffsetStrategy
                    ├─ IndexSwapStrategy
                    ├─ RandomComponentStrategy
                    └─ MixedStrategy
                            │
                            └─ Contains list of other strategies
```

## Regex Pattern Matching

```
Input Text: "Суд вирішує питання згідно з ч.2 ст.331 КПК України"
                                           └────────┬────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        │  part_article_pattern.finditer()                      │
                        └───────────────────────────┬───────────────────────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        │  Match groups:                                        │
                        │    1: "ч."     (part prefix)                          │
                        │    2: "2"      (part number)                          │
                        │    3: " ст."   (middle separator)                     │
                        │    4: "331"    (article number)                       │
                        │    5: " КПК України" (suffix with code)               │
                        └───────────────────────────┬───────────────────────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        │  strategy.modify_reference(                           │
                        │      article=331,                                     │
                        │      part=2,                                          │
                        │      code_name='КПК України',                         │
                        │      min_article=1,                                   │
                        │      max_article=615                                  │
                        │  )                                                    │
                        └───────────────────────────┬───────────────────────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        │  Returns: (333, 3, "combined_article_offset_+2_...")  │
                        └───────────────────────────┬───────────────────────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        │  Reconstruct:                                         │
                        │    "ч." + "3" + " ст." + "333" + " КПК України"       │
                        │    = "ч.3 ст.333 КПК України"                         │
                        └───────────────────────────┬───────────────────────────┘
                                                    │
                                                    ▼
Output Text: "Суд вирішує питання згідно з ч.3 ст.333 КПК України"
```

## Integration Points

```
User Code
    │
    ├────────────────────┬────────────────────┬───────────────────┐
    │                    │                    │                   │
    ▼                    ▼                    ▼                   ▼
Direct Import      CLI Interface    DocumentProcessor    graph_dataset.py
    │                    │                    │                   │
    │                    │                    │                   │
from negative_      python -m           processor =          dataset =
sampling import     negative_sampling   DocumentProcessor(   GraphDataset(
    │                    │                    │                   │
    │                    │              enable_negative_      enable_negative_
NegativeSampler()   sample-csv          sampling=True)      sampling=True)
    │                    │                    │                   │
    │                    │                    │                   │
create_negative_    sample-json        create_negative_     create_negative_
samples()           sample-bigquery    samples()            samples()
    │                    │                    │                   │
    └────────────────────┴────────────────────┴───────────────────┘
                                │
                                ▼
                    processors/__init__.py
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
        Import all        Export to         Module status
        classes           __all__           tracking
```

## File Organization

```
src/legal_graph/processors/
│
├── __init__.py                    [MODIFIED]
│   ├── Import negative_sampling classes
│   ├── Update DocumentProcessor
│   └── Export all classes
│
├── negative_sampling/             [NEW MODULE]
│   │
│   ├── __init__.py               (20KB)
│   │   ├── NegativeSampler class
│   │   ├── NegativeSampleResult model
│   │   ├── _modify_text()
│   │   ├── _modify_triplets()
│   │   ├── _modify_entities()
│   │   └── create_negative_samples()
│   │
│   ├── strategies.py             (13KB)
│   │   ├── SamplingStrategy (ABC)
│   │   ├── ArticleOffsetStrategy
│   │   ├── PartOffsetStrategy
│   │   ├── CombinedOffsetStrategy
│   │   ├── IndexSwapStrategy
│   │   ├── RandomComponentStrategy
│   │   ├── MixedStrategy
│   │   └── create_strategy()
│   │
│   ├── __main__.py               (19KB)
│   │   ├── sample_csv_data()
│   │   ├── sample_json_data()
│   │   ├── sample_bigquery_data()
│   │   └── main() CLI
│   │
│   ├── README.md                 (13KB)
│   │   └── Complete documentation
│   │
│   └── ARCHITECTURE.md           (this file)
│       └── Architecture diagrams
│
├── balance/
│   ├── __init__.py
│   │   ├── DataCorruptor (ABC)    [Parent class]
│   │   └── LegalReferenceCorruptor
│   └── __main__.py
│
├── entities/
├── graph_extraction/
└── tests/
```

## Statistics Tracking

```
NegativeSampler
    │
    ├─ corruption_stats (from DataCorruptor)
    │   ├─ total_processed: int
    │   ├─ corrupted: int
    │   ├─ skipped: int
    │   └─ errors: int
    │
    └─ sampling_stats (NegativeSampler specific)
        └─ modifications_by_type: Dict[str, int]
            ├─ "article_offset_+2": 145
            ├─ "part_offset_+1": 67
            ├─ "combined_article_offset_+3_part_offset_-1": 23
            ├─ "index_swap": 12
            └─ "random_component": 41
```

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Load Data                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │   CSV    │  │   JSON   │  │      BigQuery            │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────────────┘  │
│       └─────────────┴─────────────────┘                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  Step 2: Initialize Sampler                                 │
│  NegativeSampler(strategy='mixed', probability=0.5)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  Step 3: Process Each Sample                                │
│  for sample in dataset:                                     │
│      if random() < probability:                             │
│          result = corrupt_sample(sample)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  Step 4: Detect & Modify References                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │  Detect      │→ │  Extract     │→ │  Apply         │    │
│  │  patterns    │  │  components  │  │  strategy      │    │
│  └──────────────┘  └──────────────┘  └────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  Step 5: Label & Track                                      │
│  • Add is_valid label (True/False)                          │
│  • Track modification types                                 │
│  • Update statistics                                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  Step 6: Output                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │   CSV    │  │   JSON   │  │      BigQuery            │  │
│  └──────────┘  └──────────┘  └──────────────────────────┘  │
│  + Statistics report                                        │
└─────────────────────────────────────────────────────────────┘
```

## Legal Reference Patterns

```
Ukrainian Legal Reference Patterns

Full Pattern:  ч.X ст.Y КК України
    └─┬─┘ └─┬─┘ └──┬──┘ └────┬────┘
   part  art.    code    country

Components:
• ч. (частина) = Part
• ст. (стаття) = Article
• КК = Criminal Code (Кримінальний Кодекс)
• України = of Ukraine

Variations:
1. ч.2 ст.185 КК України      (full form - MOST COMMON)
2. ст.185 КК України          (without part)
3. ст.185 ч.2 КК України      (reversed order)
4. п.2 ст.185 КК України      (п. instead of ч.)

Legal Codes:
• КК України   (Criminal Code)           1-446 articles
• КПК України  (Criminal Procedure)      1-615 articles
• ЦК України   (Civil Code)              1-1308 articles
• КоАП України (Administrative)          1-328 articles
• СК України   (Family Code)             1-270 articles
• КЗпП України (Labor Code)              1-265 articles
```

## Performance Considerations

```
Processing Speed Factors:

1. Regex Matching
   • Compiled patterns (pre-compiled in __init__)
   • Reverse iteration to maintain positions
   • Single-pass processing where possible

2. Strategy Selection
   • O(1) strategy lookup via factory pattern
   • Pre-instantiated default strategies
   • Minimal overhead for strategy switching

3. Batch Processing
   • CLI supports batch_size parameter
   • BigQuery pagination for large datasets
   • Memory-efficient streaming

4. Statistics
   • Incremental updates (not recalculated)
   • Dictionary lookups O(1)
   • Minimal memory footprint

Estimated Performance:
• ~100-500 documents/second (CSV/JSON)
• ~50-200 documents/second (BigQuery with network)
• Scales linearly with document count
```

This architecture provides a robust, extensible, and well-integrated solution for creating hard negative samples in the TopoLegal system.
