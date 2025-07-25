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
        self.llm = OllamaLLM(model=model_name)

        self.parser = PydanticOutputParser(pydantic_object=SummaryOutput)
        self.prompt = ChatPromptTemplate.from_template("""
        Вот таблица в формате markdown со статистикой анализа отзывов по категориям и тональностям: {stats}.

        Сделай краткое саммари на русском языке: какие категории чаще всего упоминаются, и опиши каждую категорию по тональности.
        Ответ формируй исходя только из данных в таблице, не добавляй ничего лишнего.
                                                       
        Формат ответа должен быть таким:                                     
        {format_instructions}
        """)
        self.final_prompt = self.prompt.partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.final_prompt | self.llm | self.parser

        self.aspect_analyser = AspectAnalyser(model_name=model_name)


    def stats_to_df(self, stats):
        """
        Convert the statistics dictionary to a pandas DataFrame.

        Args:
            stats (dict): A dictionary containing statistics.

        Returns:
            pd.DataFrame: A DataFrame representation of the statistics.
        """
        self.sentiments = ["положительный", "отрицательный", "нейтральный"]
        rows = []
        for category, values in stats.items():
            row = {'category': category, 'всего': values['count']}
            for sentiment in self.sentiments:
                row[sentiment] = values['sentiments'].get(sentiment, 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df[['category', 'положительный', 'отрицательный', 'нейтральный', 'всего']]

        return df
    

    def summarize(self, reviews):
        analysys_results = self.aspect_analyser.full_analysis(reviews)
        stats_str = self.stats_to_df(analysys_results['stats']).to_markdown(index=False)

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
        "Сервис ужасный, больше не приду.",
        "Доставка быстрая."
    ]

    s = Summarizer()
    res = s.summarize(reviews)
    print(res)