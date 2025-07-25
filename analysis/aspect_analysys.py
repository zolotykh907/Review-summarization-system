from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from pydantic import ValidationError

import pandas as pd


from typing import List


class CategorySentiment(BaseModel):
    category: str = Field(..., description="Категория, к которой относится отзыв")
    sentiment: str = Field(..., description="Тональность для этой категории")
    #description: str = Field(..., description="Одно или два слова, что пишут про эту категорию в отзыве, на русском языке")


class AspectOutput(BaseModel):
    aspects: List[CategorySentiment] = Field(..., description="Список категорий и соответствующих тональностей")


class AspectAnalyser:
    def __init__(self, model_name='llama3'):
        self.categories = ["товар", "обслуживание", "доставка", "цена", "качество", "интерфейс", "другое"]
        self.sentiments = ["положительный", "нейтральный", "отрицательный"]
        self.parser = PydanticOutputParser(pydantic_object=AspectOutput)
        self.template = """
        Ты — эксперт по анализу отзывов. Проанализируй следующий отзыв и определи, какие из **предопределённых категорий** в нём явно упоминаются. 
        Отзыв может относиться к нескольким категориям одновременно.

        Категории выбирай строго из списка: {categories}
        
        Так же определи для каждой найденно категории определи тональность.
        Тональности выбирай строго из списка: {sentiments}
        
        Отзыв: {review}


        Формат ответа должен быть таким:
        {format_instructions}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = OllamaLLM(model=model_name, temperature=0.1, max_tokens=1000)
        self.final_prompt = self.prompt.partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.final_prompt | self.llm | self.parser


    def aspect_analysis(self, reviews, categories, sentiments):
        """
        Analyze the aspects of a reviews.

        Args:
            reviews (list): a list of review texts to analyze.
            categories (list): a list of categories to choose from.
            sentiments (list): a list of sentiments to choose from.

        Returns:
            list: a list of aspect analysis results for each review.
        """
        results = []
        for review in reviews:
            try: 
                response = self.chain.invoke({"review": review, 
                                              "categories": categories, 
                                              "sentiments": sentiments,
                                              "format_instructions": self.parser.get_format_instructions()})
                results.append(response)
            except ValidationError as e:
                raise ValueError(f"Ошибка валидации: {e}")
            
        return results
    

    def stats_analysis(self, analysis_results):
        """
        Analyze statistics from the aspect analysis results.

        Args:
            analysis_results (list): a list of aspect analysis results.

        Returns:
            dict: a dictionary with counts of each category and sentiment.
        """
        stats = {}

        for result in analysis_results:
            for aspect in result.aspects:
                category = aspect.category.lower()
                sentiment = aspect.sentiment.lower()
                
                if category not in stats:
                    stats[category] = {"count": 0, "sentiments": {}}
                stats[category]["count"] += 1
                
                if sentiment not in stats[category]["sentiments"]:
                    stats[category]["sentiments"][sentiment] = 0
                stats[category]["sentiments"][sentiment] += 1

                #stats[category]["description"] = aspect.description

        return stats
    

    def full_analysis(self, reviews):
        """
        Full analysis of a list of reviews, including aspects and statistics.

        Args:
            reviews (list): a list of review texts to analyze.
            categories (list): a list of categories to choose from.
            sentiments (list): a list of sentiments to choose from.

        Returns:
            dict: a dictionary with aspects and statistics.
        """
        aspects = self.aspect_analysis(reviews, self.categories, self.sentiments)
        stats = self.stats_analysis(aspects)

        return {
            "aspects": aspects,
            "stats": stats
        }
    

if __name__ == "__main__":
    # Пример использования
    reviews = [
        "Отличный товар, очень доволен покупкой!",
        "Доставка была задержана",
        "Доставка ужасно долгая, но обслуживание отличное.",
    ]
    
    categories = ["товар", "обслуживание", "доставка", "цена", "качество", "интерфейс", "другое"]
    sentiments = ["положительный", "нейтральный", "отрицательный"]
    
    analyzer = AspectAnalyser()
    results = analyzer.aspect_analysis(reviews, categories, sentiments)
    stats = analyzer.stats_analysis(results)
    
    print(stats)