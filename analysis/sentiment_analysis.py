import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser


class SentimentOutput(BaseModel):
    sentiment: str = Field(description="Sentiment label: positive, neutral or negative")
    #explanation: str = Field(description="Explanation in Russian")


class SentimentAnalyser:
    def __init__(self, model_name='llama3'):
        self.template = """
        Проанализируй тональность этого отзыва: {review}\n

        Ответ должен быть строго в формате JSON:
        {{
        "sentiment": "<positive|neutral|negative>",
        }}
        """
        self.llm = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(template=self.template)
        self.parser = PydanticOutputParser(pydantic_object=SentimentOutput)
        self.final_prompt = self.prompt.partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.final_prompt | self.llm | self.parser


    def sentiment_analysis(self, reviews):
        """
        Analyze the sentiment of a reviews.
        
        Args:
            reviews (list): a list of review texts to analyze.
        
        Returns:
            list: a list of sentiments for each review.
        """
        results = []
        for review in reviews:
            try:
                response = self.chain.invoke({"review": review})
                results.append(response)
            except ValidationError as e:
                raise ValueError(f"Error validation: {e}")
        return results
    

    def stats_analysis(self, analysis_results):
        """
        Analyze statistics from the sentiment analysis results.
        
        Args:
            analyze_results (list): a list of sentiment results.
        
        Returns:
            dict: a dictionary with counts of each sentiment type.
        """

        stats = {"positive": 0, "negative": 0, "neutral": 0}
        for result in analysis_results:
            if result:
                stats[result.sentiment] += 1
        return stats
    

    def full_analysis(self, reviews):
        """
        Full analysis of a list of reviews, including sentiment and statistics.
        
        Args:
            reviews (list): a list of review texts to analyze.
        
        Returns:
            dict: a dictionary with sentiments and statistics.
        """
        analysys_results = self.sentiment_analysis(reviews)
        stats = self.stats_analysis(analysys_results)
        return {
            "sentiments": analysys_results,
            "statistics": stats
        }
    

if __name__ == "__main__":
    # Example usage
    reviews = [
        "Отличный продукт, очень доволен!",
        "Сервис ужасный, больше не приду.",
        "Нормально, но можно лучше."
    ]

    A = SentimentAnalyser()
    results = A.full_analysis(reviews)

    print("Sentiments:", results['sentiments'])
    print("Statistics:", results['statistics'])