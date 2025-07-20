import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser


class SentimentOutput(BaseModel):
    sentiment: str = Field(description="Sentiment label: positive, neutral or negative")
    explanation: str = Field(description="Explanation in Russian")


class SentimentAnalyser:
    def __init__(self, model_name='llama3'):
        self.template = """
        Проанализируй тональность этого отзыва: {review}\n

        Ответ должен быть строго в формате JSON:
        {{
        "sentiment": "<positive|neutral|negative>",
        "explanation": "<текст>"
        }}
        
        Объяснение обязательно должно быть на русском языке.
        """
        self.llm = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(template=self.template)
        self.parser = PydanticOutputParser(pydantic_object=SentimentOutput)
        self.final_prompt = self.prompt.partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.final_prompt | self.llm | self.parser


    def analyze_sentiment(self, review):
        """
        Analyze the sentiment of a review.
        
        Args:
            review (str): the review text to analyze.
        
        Returns:
            str: the sentiment of the review (positive, negative, neutral).
        """
        try:
            response = self.chain.invoke({"review": review})
            return response
        except ValidationError as e:
            raise ValueError(f"Error validation: {e}")


    def analyze_sentiment_list(self, reviews):
        """
        Analyze the sentiment of a list of reviews.
        
        Args:
            reviews (list): a list of review texts to analyze.
        
        Returns:
            list: a list of sentiments for each review.
        """
        results = []
        for review in reviews:
            res = self.analyze_sentiment(review)
            results.append(res)
        return results
    

    def analyze_stats(self, analyze_results):
        """
        Analyze statistics from the sentiment analysis results.
        
        Args:
            analyze_results (list): a list of sentiment results.
        
        Returns:
            dict: a dictionary with counts of each sentiment type.
        """

        stats = {"positive": 0, "negative": 0, "neutral": 0}
        for result in analyze_results:
            if result:
                stats[result.sentiment] += 1
        return stats
    

#Usage example:
reviews = [
    "Отличный продукт, очень доволен!",
    "Сервис ужасный, больше не приду.",
    "Нормально, но можно лучше."
]

A = SentimentAnalyser()
results = A.analyze_sentiment_list(reviews)
stats = A.analyze_stats(results)

print(f"Количество обработанных отзывов: {len(results)}")
print(f"Статистика тональности: {stats}")