from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain

class KnowledgeTriplet(BaseModel):
    """Represents a knowledge graph triplet (source, relation, target)"""
    source: str
    relation: str
    target: str
    legal_reference: Optional[str] = None  # Legal basis for this relationship
    confidence: float = 0.0

    def __str__(self) -> str:
        return f"({self.source}, {self.relation}, {self.target})"

class LegalEntity(BaseModel):
    """Enhanced entity with legal context"""
    text: str
    label: str  # NER label (PERSON, ORG, ROLE, etc.)
    start: int
    end: int
    legal_role: Optional[str] = None  # Role in legal context (підозрюваний, суд, etc.)

class LegalKnowledgeGraph(BaseModel):
    """Knowledge graph represented as triplets with legal metadata"""
    triplets: List[KnowledgeTriplet]
    entities: List[LegalEntity]
    legal_references: List[str]  # All legal references found in text

    def get_triplets_by_legal_ref(self, legal_ref: str) -> List[KnowledgeTriplet]:
        """Get all triplets associated with a specific legal reference"""
        return [t for t in self.triplets if t.legal_reference == legal_ref]

    def get_entity_relations(self, entity: str) -> List[KnowledgeTriplet]:
        """Get all relations where entity appears as source or target"""
        return [t for t in self.triplets if t.source == entity or t.target == entity]

class LegalGraphExtractor:
    def __init__(self, model_name: str = "gpt-4.1-nano", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["text", "entities"],
            template="""На основі наданого тексту та виявлених сутностей, створіть граф знань у форматі триплетів (джерело, відношення, ціль).

Текст: {text}

Виявлені сутності: {entities}

Приклад аналізу:
Текст: "Відповідно до положень ч.2 ст.331 КПК України, суд вирішує питання про запобіжний захід. Підозрюваному може бути призначено домашній арешт згідно з ч.1 ст.181 КПК України."

Очікувані триплети:
1. (суд, вирішує_питання_про, запобіжний захід) - ч.2 ст.331 КПК України
2. (підозрюваному, може_бути_призначено, домашній арешт) - ч.1 ст.181 КПК України
3. (запобіжний захід, регулюється, ч.2 ст.331 КПК України)
4. (домашній арешт, регулюється, ч.1 ст.181 КПК України)
5. (суд, має_повноваження, призначення запобіжного заходу)

Формат виводу як JSON:
{{
    "triplets": [
        {{
            "source": "string",
            "relation": "string", 
            "target": "string",
            "legal_reference": "string",
            "confidence": float
        }}
    ],
    "legal_references": ["string"]
}}

Правила створення триплетів:
1. Використовуйте конкретні сутності з тексту як джерела та цілі
2. Відношення повинні відображати правові зв'язки (призначає, регулюється, підлягає, застосовується)
3. Обов'язково вказуйте правове посилання для кожного триплету
4. Створюйте додаткові триплети для зв'язків між правовими нормами та процедурами
5. Використовуйте українські терміни у відношеннях

Зосередьтеся на:
- Процедурних відношеннях (хто що робить)
- Правових основах (що чим регулюється)
- Суб'єктно-об'єктних відношеннях у правовому процесі"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def extract_triplets(self, text: str, entities: List[Dict]) -> LegalKnowledgeGraph:
        """Extract knowledge graph triplets from legal text and entities"""
        # Convert entities to LegalEntity objects
        legal_entities = [
            LegalEntity(
                text=e.get("text", ""),
                label=e.get("label", ""),
                start=e.get("start", 0),
                end=e.get("end", 0),
                legal_role=self._determine_legal_role(e.get("text", ""), e.get("label", ""))
            ) for e in entities
        ]

        # Extract triplets using LLM
        result = await self.chain.arun(text=text, entities=str(entities))

        try:
            import json
            parsed_result = json.loads(result)

            triplets = [
                KnowledgeTriplet(**triplet_data)
                for triplet_data in parsed_result.get("triplets", [])
            ]

            legal_references = parsed_result.get("legal_references", [])

            return LegalKnowledgeGraph(
                triplets=triplets,
                entities=legal_entities,
                legal_references=legal_references
            )

        except json.JSONDecodeError:
            # Fallback: create empty graph
            return LegalKnowledgeGraph(
                triplets=[],
                entities=legal_entities,
                legal_references=[]
            )

    def _determine_legal_role(self, text: str, label: str) -> Optional[str]:
        """Determine legal role based on entity text and NER label"""
        legal_roles_map = {
            "суд": "judicial_authority",
            "підозрюваний": "suspect",
            "обвинувачений": "accused",
            "прокурор": "prosecutor",
            "адвокат": "defense_attorney",
            "слідчий": "investigator",
            "потерпілий": "victim"
        }

        text_lower = text.lower()
        for term, role in legal_roles_map.items():
            if term in text_lower:
                return role

        return None

    def validate_triplets(self, kg: LegalKnowledgeGraph) -> List[str]:
        """Validate extracted triplets for legal consistency"""
        issues = []

        for triplet in kg.triplets:
            # Check if legal reference is provided
            if not triplet.legal_reference:
                issues.append(f"Missing legal reference for: {triplet}")

            # Check confidence threshold
            if triplet.confidence < 0.5:
                issues.append(f"Low confidence for: {triplet}")

            # Check for empty relations
            if not triplet.relation.strip():
                issues.append(f"Empty relation in: {triplet}")

        return issues

    def export_to_networkx(self, kg: LegalKnowledgeGraph):
        """Export knowledge graph to NetworkX format for analysis"""
        try:
            import networkx as nx

            G = nx.DiGraph()

            for triplet in kg.triplets:
                G.add_edge(
                    triplet.source,
                    triplet.target,
                    relation=triplet.relation,
                    legal_reference=triplet.legal_reference,
                    confidence=triplet.confidence
                )

            return G

        except ImportError:
            raise ImportError("NetworkX is required for graph export. Install with: pip install networkx")

# Usage example
async def process_legal_document(text: str, entities: List[Dict]) -> LegalKnowledgeGraph:
    """Process a legal document and extract knowledge graph from pre-extracted entities"""
    extractor = LegalGraphExtractor()
    kg = await extractor.extract_triplets(text, entities)

    # Validate results
    issues = extractor.validate_triplets(kg)
    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(f"- {issue}")

    # Print extracted triplets
    print(f"\nExtracted {len(kg.triplets)} triplets:")
    for triplet in kg.triplets:
        legal_ref = f" [{triplet.legal_reference}]" if triplet.legal_reference else ""
        print(f"  {triplet}{legal_ref}")

    return kg