from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser

from categoty_analysis import CategoryAnalyser
from sentiment_analysis import SentimentAnalyser
from aspect_analysys import AspectAnalyser


class SummaryOutput(BaseModel):
    summary: str = Field(description="Текст итогового резюме на русском языке")


class Summarizer:
    def __init__(self, model_name='llama3'):
        self.llm = OllamaLLM(model=model_name)

        self.parser = PydanticOutputParser(pydantic_object=SummaryOutput)
        format_instructions = self.parser.get_format_instructions()

        self.prompt = ChatPromptTemplate.from_template("""
        Сделай обобщённое резюме по следующим отзывам:
        {reviews}

        Для каждого отзыва заранее определены:
        - Тональность: {sentiments}
        - Категория: {categories}

        Сформируй краткое обобщение на русском языке. Используй следующую инструкцию по формату:

        {format_instructions}
        """)
        self.chain = self.prompt | self.llm | self.parser

        self.aspect_analyser = AspectAnalyser(model_name=model_name)

    def summarize(self, reviews):
        sentiments = self.sentiment_analyser.sentiment_analysis(reviews)
        categories = self.category_analyser.category_analysys(reviews)

        try:
            response = self.chain.invoke({
                "reviews": reviews,
                "sentiments": [s.sentiment for s in sentiments],
                "categories": [c.category for c in categories],
                "format_instructions": self.parser.get_format_instructions()
            })
            return response
        except ValidationError as e:
            raise ValueError(f"Ошибка валидации: {e}")

                         
if __name__ == "__main__":
    # Пример использования
    reviews = [
        "Отличный продукт, очень доволен!",
        "Сервис ужасный, больше не приду.",
        "Доставка быстрая."
    ]

    s = Summarizer()
    res = s.summarize(reviews)
    print(res)