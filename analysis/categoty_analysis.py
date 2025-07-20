import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser


class CategoryOutput(BaseModel):
    category: Literal[
        "товар", "обслуживание", "доставка", "цена", "качество", "интерфейс", "другое"
    ] = Field(description="Категория отзыва")

class CategoryAnalyser():
    def __init__(self, model_name='llama3'):
        self.categories = [
            "товар", "обслуживание", "доставка", "цена", "качество", "интерфейс", "другое"
        ]
        self.template = """
        Определи категорию этого отзыва: {review}\n
        
        Вот список доступных категорий:
        {categories}\n

        Если ничего из этого не подходит, выбери категорию "другое".

        Ответ должен быть строго в формате JSON:
        {{"category": "<выбранная категория>"}}.
        """

        self.llm = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.parser = PydanticOutputParser(pydantic_object=CategoryOutput)
        self.final_prompt = self.prompt.partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.final_prompt | self.llm | self.parser

    def category_analysys(self, reviews):
        """
        Analyze the category of a reviews.
        
        Args:
            reviews (list): a list of review texts to analyze.
        
        Returns:
            list: a list of categories for each review.
        """
        results = []
        for review in reviews:
            try:
                response = self.chain.invoke({"categories": self.categories, "review": review})
                results.append(response)
            except ValidationError as e:
                raise ValueError(f"Ошибка валидации: {e}")

        return results
    
    
    def stats_analysis(self, analysis_results):
        """
        Analyze statistics from the category analysis results.
        
        Args:
            analyze_results (list): a list of category results.
        
        Returns:
            dict: a dictionary with counts of each category.
        """
        stats = {category: 0 for category in self.categories}
        for result in analysis_results:
            stats[result.category] += 1
        return stats
    

    def full_analysis(self, reviews):
        """
        Full analysis of a list of reviews, including categories and statistics.
        
        Args:
            reviews (list): a list of review texts to analyze.
        
        Returns:
            dict: a dictionary with categories and statistics.
        """
        categories = self.category_analysys(reviews)
        stats = self.stats_analysis(categories)
        
        return {
            "categories": categories,
            "statistics": stats
        }


#Usage example
# reviews = [
#     "Отличный продукт, очень доволен!",
#     "Сервис ужасный, больше не приду.",
#     "Доставка быстрая."
# ]

# C = CategoryAnalyser()
# res = C.full_analysis(reviews)

# print(res['categories'])
# print(res['statistics'])