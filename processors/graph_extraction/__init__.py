from typing import Dict, List, Optional
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from processors import DocumentProcessor

class EventNode(BaseModel):
    """Represents a node in the event graph"""
    event_type: str
    legal_reference: str  # Legal reference (e.g., "ч.2 ст.331 КПК України")
    participants: List[Dict[str, str]]
    attributes: Dict[str, str]
    confidence: float

class EventEdge(BaseModel):
    """Represents an edge in the event graph"""
    source: str
    target: str
    relation_type: str
    legal_basis: Optional[str]  # Legal reference that establishes the relationship
    confidence: float

class EventGraph(BaseModel):
    """Represents the complete event graph"""
    nodes: List[EventNode]
    edges: List[EventEdge]

class GraphExtractor:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["text", "entities"],
            template="""На основі наданого тексту та виявлених сутностей, визначте події та їх взаємозв'язки, прив'язуючи кожну подію до відповідного правового посилання.

Текст: {text}

Виявлені сутності: {entities}

Приклад аналізу судового документа:
Текст: "Відповідно до положень ч.2 ст.331 КПК України, вирішення питання судом щодо запобіжного заходу відбувається в порядку, передбаченому главою 18 цього Кодексу. Згідно з ч.ч.1, 2 ст.181 КПК України, запобіжний захід у вигляді домашнього арешту полягає в забороні підозрюваному, обвинуваченому залишати житло цілодобово."

Сутності: [
    {{"text": "суд", "label": "ORG", "start": 50, "end": 54}},
    {{"text": "підозрюваному", "label": "ROLE", "start": 120, "end": 133}},
    {{"text": "обвинуваченому", "label": "ROLE", "start": 135, "end": 149}}
]

Очікуваний результат:
{{
    "nodes": [
        {{
            "event_type": "ВИРІШЕННЯ_ПИТАННЯ_ЗАПОБІЖНОГО_ЗАХОДУ",
            "legal_reference": "ч.2 ст.331 КПК України",
            "participants": [
                {{"role": "суб'єкт", "entity": "суд"}}
            ],
            "attributes": {{
                "порядок": "глава 18 КПК України"
            }},
            "confidence": 0.95
        }},
        {{
            "event_type": "ЗАСТОСУВАННЯ_ДОМАШНЬОГО_АРЕШТУ",
            "legal_reference": "ч.ч.1, 2 ст.181 КПК України",
            "participants": [
                {{"role": "суб'єкт", "entity": "підозрюваному"}},
                {{"role": "суб'єкт", "entity": "обвинуваченому"}}
            ],
            "attributes": {{
                "тип_обмеження": "заборона залишати житло цілодобово"
            }},
            "confidence": 0.95
        }}
    ],
    "edges": [
        {{
            "source": "ВИРІШЕННЯ_ПИТАННЯ_ЗАПОБІЖНОГО_ЗАХОДУ",
            "target": "ЗАСТОСУВАННЯ_ДОМАШНЬОГО_АРЕШТУ",
            "relation_type": "ПРАВОВА_ПОСЛІДОВНІСТЬ",
            "legal_basis": "ч.2 ст.331 КПК України",
            "confidence": 0.95
        }}
    ]
}}

Будь ласка, визначте:
1. Події (вузли) з їх типом, правовим посиланням, учасниками та атрибутами
2. Взаємозв'язки (ребра) між подіями з вказанням правової підстави

Формат виводу як JSON з наступною структурою:
{{
    "nodes": [
        {{
            "event_type": "string",
            "legal_reference": "string",
            "participants": [{{"role": "string", "entity": "string"}}],
            "attributes": {{"key": "value"}},
            "confidence": float
        }}
    ],
    "edges": [
        {{
            "source": "string",
            "target": "string",
            "relation_type": "string",
            "legal_basis": "string",
            "confidence": float
        }}
    ]
}}

Зосередьтеся на:
1. Визначенні подій, пов'язаних з правовими посиланнями
2. Встановленні зв'язків між подіями на основі правових норм
3. Структуруванні подій для подальшої валідації через GNN модель"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def extract_events(self, text: str, entities: List[Dict]) -> EventGraph:
        """Extract events and their relationships from text and entities"""
        result = await self.chain.arun(text=text, entities=str(entities))
        return EventGraph.parse_raw(result) 