from typing import Dict, List, Optional
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from processors import DocumentProcessor

class EventNode(BaseModel):
    """Represents a node in the event graph"""
    event_type: str
    participants: List[Dict[str, str]]
    attributes: Dict[str, str]
    confidence: float

class EventEdge(BaseModel):
    """Represents an edge in the event graph"""
    source: str
    target: str
    relation_type: str
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
            template="""На основі наданого тексту та виявлених сутностей, визначте події та їх взаємозв'язки для формування графової структури.

Текст: {text}

Виявлені сутності: {entities}

Приклад аналізу судового документа:
Текст: "Справа № 947/37275/21. Слідчий суддя Київського районного суду міста Одеси ОСОБА_1 розглянувши клопотання про обрання запобіжного заходу у вигляді тримання під вартою відносно ОСОБА_9. 24.11.2021 року надійшло клопотання. 17.12.2021 року було подано змінене клопотання."

Сутності: [
    {{"text": "ОСОБА_1", "label": "PERSON", "start": 50, "end": 58}},
    {{"text": "ОСОБА_9", "label": "PERSON", "start": 120, "end": 128}},
    {{"text": "24.11.2021", "label": "DATE", "start": 150, "end": 160}},
    {{"text": "17.12.2021", "label": "DATE", "start": 180, "end": 190}}
]

Очікуваний результат:
{{
    "nodes": [
        {{
            "event_type": "ПОДАННЯ_КЛОПОТАННЯ",
            "participants": [
                {{"role": "заявник", "entity": "ОСОБА_1"}},
                {{"role": "обвинувачений", "entity": "ОСОБА_9"}}
            ],
            "attributes": {{
                "дата": "24.11.2021",
                "тип_заходу": "тримання під вартою"
            }},
            "confidence": 0.95
        }},
        {{
            "event_type": "ЗМІНА_КЛОПОТАННЯ",
            "participants": [
                {{"role": "заявник", "entity": "ОСОБА_1"}},
                {{"role": "обвинувачений", "entity": "ОСОБА_9"}}
            ],
            "attributes": {{
                "дата": "17.12.2021",
                "підстава": "ч.2 ст.185 КПК України"
            }},
            "confidence": 0.95
        }}
    ],
    "edges": [
        {{
            "source": "ПОДАННЯ_КЛОПОТАННЯ",
            "target": "ЗМІНА_КЛОПОТАННЯ",
            "relation_type": "ЗМІНА_ДОКУМЕНТУ",
            "confidence": 0.95
        }}
    ]
}}

Будь ласка, визначте:
1. Події (вузли) з їх типом, учасниками та атрибутами
2. Взаємозв'язки (ребра) між подіями

Формат виводу як JSON з наступною структурою:
{{
    "nodes": [
        {{
            "event_type": "string",
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
            "confidence": float
        }}
    ]
}}

Зосередьтеся на юридичних подіях та їх взаємозв'язках, використовуючи методологію з CrossRE."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    async def extract_events(self, text: str, entities: List[Dict]) -> EventGraph:
        """Extract events and their relationships from text and entities"""
        result = await self.chain.arun(text=text, entities=str(entities))
        return EventGraph.parse_raw(result) 