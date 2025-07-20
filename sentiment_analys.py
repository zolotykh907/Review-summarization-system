from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


class SentimentAnalyser:
    def __init__(self, model_name='llama3'):
        self.template = """
        Проанализируй тональность этого отзыва: {review}\n

        Важно: ответ должен быть в формате JSON:
        {{sentiment: "positive" | "negative" | "neutral", 
        explanation: "Здесь должно быть краткое объяснение твоего решения"}}
        
        Объяснение обязательно должно быть на русском языке.
        """
        self.llm = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(template=self.template)
        self.chain = self.prompt | self.llm


    def get_analyze_dict(self, analyze_result):
        """
        Convert the analysis result into a dictionary.
        
        Args:
            analyze_result (str): the result of the sentiment analysis.
        
        Returns:
            dict: a dictionary with sentiment and explanation.
        """
        analyze_result = analyze_result.strip()
        if 'sentiment' in analyze_result:
            sentiment = analyze_result.split('sentiment: ')
            return sentiment


    def analyze_sentiment(self, review):
        """
        Analyze the sentiment of a review.
        
        Args:
            review (str): the review text to analyze.
        
        Returns:
            str: the sentiment of the review (positive, negative, neutral).
        """
        response = self.chain.invoke({"review": review})
        #return response
        return self.get_analyze_dict(response)
    

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
    

    def analyze_statistics(self, analyze_results):
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
                stats[result] += 1
        return stats

A = SentimentAnalyser()
print(A.analyze_sentiment("Отличный продукт, очень доволен!"))