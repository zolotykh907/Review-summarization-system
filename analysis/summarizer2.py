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
    

    def combine_stats(self, categories, sentiments):
        stats = {}
        for i in range(len(categories)):
            if categories[i] not in stats:
                stats[categories[i]] = {sentiment: 0 for sentiment in self.sentiments}
            stats[categories[i]][sentiments[i]] += 1

        return stats


    def summarize_stats(self, reviews):
        categories = self.category_analyser.full_analysis(reviews)['categories']
        sentiments = self.sentiment_analyser.full_analysis(reviews)['sentiments']

        categories = [c.category for c in categories]
        sentiments = [s.sentiment for s in sentiments]

        combined_stats = self.combine_stats(categories, sentiments)
        stats_str = self.stats_to_text(combined_stats)

        try:
            response = self.chain.invoke({
                "stats": stats_str,
                "format_instructions": self.parser.get_format_instructions()
            })
            return response.model_dump()
        except ValidationError as e:
            raise ValueError(f'Validation Error: {e}')
    

    def split_to_batches(self, reviews):
        self.batch_size = 5

        batches = []
        for i in range(0, len(reviews), self.batch_size):
            batches.append(reviews[i: i+self.batch_size])

        return batches
    

    def summarize_batches(self, reviews):
        batch_prompt = ChatPromptTemplate.from_template("""
        Ты — эксперт по анализу отзывов. Проанализируй следующие отзывы и дай обобщенное краткое резюме на 2-3 предложения.
        Отвечай строго на русском языке.
                                                        
        Отзывы: {reviews}
        """)
        batch_chain = batch_prompt | self.llm
        
        batches = self.split_to_batches(reviews)
        results = []
        for batch in batches:
            try:
                response = batch_chain.invoke({"reviews": "\n".join(batch)})
                results.append(response)
            except ValidationError as e:
                raise ValueError(f'Validation Error: {e}')
        return results
        

    def summarize_reviews(self, reviews):
        summary_prompt = ChatPromptTemplate.from_template("""
        Ты - эксперт по анализу отзывов. Вот краткая сводка по отзывам: 
                                                          
        {summary_text}
                                                          
        Твоя задача: дай краткое резюме на 3-5 предложений, подводя итоги по всем отзывам.
                                                          
        Отвечай строго на русском языке.                                             
        """)
        summary_chain = summary_prompt | self.llm
        summary_batches = self.summarize_batches(reviews)
        summary_text = "\n".join(summary_batches)

        try:
            response = summary_chain.invoke({"summary_text": summary_text})
            return response
        except ValidationError as e:
            raise ValueError(f'Validation Error: {e}')
        
    
    def get_recommendation(self, summary_results):
        recommendation_prompt = ChatPromptTemplate.from_template("""
        Ты - эксперт по анализу отзывов. Вот краткая сводка по отзывам:

        {summary_results}                                       
        
        Нужно написать рекомендации по улучшению того, о чем говорится в этой сводке.
        Не пиши ничего лишнего, только про те вещи, которые имеют минусы.

        Отвечай строго на русском языке.                                        
        """)
        recommendation_chain = recommendation_prompt | self.llm

        try:
            response = recommendation_chain.invoke({"summary_results": summary_results})
            return response
        except ValidationError as e:
            raise ValueError(f'Validation Error: {e}')

                         
if __name__ == "__main__":
    # Usage example
    reviews = [
        "Отличный продукт, очень доволен!",
        "Доставка быстрая.",
        "В принципе, неплохо, но есть недочеты.",
        "Обслуживание на высоте",
        "Качество товара оставляет желать лучшего",
    ]

    s = Summarizer()
    # res = s.summarize_stats(reviews)
    # print(res)

    summary = s.summarize_reviews(reviews)
    print(summary)
    print(s.get_recommendation(summary_results=summary))