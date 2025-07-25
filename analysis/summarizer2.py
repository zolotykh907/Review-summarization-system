from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser
import pandas as pd

from categoty_analysis import CategoryAnalyser
from sentiment_analysis import SentimentAnalyser
from aspect_analysys import AspectAnalyser


class SummaryOutput(BaseModel):
    summary: str = Field(..., description="Текст итогового резюме на русском языке")


class Summarizer:
    def __init__(self, model_name='llama3'):
        self.sentiments = ["позитивный", "нейтральный", "негативный"]
        self.llm = OllamaLLM(model=model_name, temperature=0.1)

        self.parser = PydanticOutputParser(pydantic_object=SummaryOutput)
    
        self.prompt = ChatPromptTemplate.from_template("""
        Ниже представлена статистика анализа отзывов. 
        По каждой категории указано, сколько отзывов с каждой тональностью.
        Отвечай на русском языке.

        Твоя задача:
        1. Определи, какие категории упоминаются чаще всего.
        2. Для каждой категории опиши, какой тип отзывов преобладает — положительный, нейтральный или отрицательный.
        3. Не делай выводов вне представленных чисел. Формулировки должны быть краткими и объективными.
        4. Напиши только {{'summary': <текст>}}

        Статистика:
        {stats}

        Формат вывода (JSON):
        {format_instructions}
        """)
        self.final_prompt = self.prompt.partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.final_prompt | self.llm | self.parser

        self.aspect_analyser = AspectAnalyser(model_name=model_name)
        self.category_analyser = CategoryAnalyser(model_name=model_name)
        self.sentiment_analyser = SentimentAnalyser(model_name=model_name)
    

    def stats_to_text(self, stats: dict) -> str:
        lines = []
        for category, sentiments in stats.items():
            lines.append(
                f"Категория: {category}\n"
                f"Позитивных: {sentiments['позитивный']}\n"
                f"Негативных: {sentiments['негативный']}\n"
                f"Нейтральных: {sentiments['нейтральный']}\n"
            )
        return "\n".join(lines)


    def summarize(self, reviews):
        categories = self.category_analyser.full_analysis(reviews)['categories']
        sentiments = self.sentiment_analyser.full_analysis(reviews)['sentiments']

        categories = [c.category for c in categories]
        sentiments = [s.sentiment for s in sentiments]

        stats = {}
        for i in range(len(categories)):
            if categories[i] not in stats:
                stats[categories[i]] = {sentiment: 0 for sentiment in self.sentiments}
            stats[categories[i]][sentiments[i]] += 1

        stats_str = self.stats_to_text(stats)

        try:
            response = self.chain.invoke({
                "stats": stats_str,
                "format_instructions": self.parser.get_format_instructions()
            })
            return response.model_dump()
        except ValidationError as e:
            raise ValueError(f"Ошибка валидации: {e}")
        

                         
if __name__ == "__main__":
    # Usage example
    reviews = [
        "Отличный продукт, очень доволен!",
        "Доставка ужасная",
        "Доставка быстрая."
    ]

    s = Summarizer()
    res = s.summarize(reviews)
    print(res)