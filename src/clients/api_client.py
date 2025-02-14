"""
Unified API client for handling all external API interactions.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import google.cloud.language as google_language
from openai import OpenAI
import google.generativeai as genai
import httpx
from functools import lru_cache

# Load environment variables
load_dotenv()

class APIClient:
    def __init__(self):
        # Initialize API keys
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.google_sentiment_key = os.getenv('GOOGLE_SENTIMENT_API_KEY')
        self.perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        
        # Initialize clients
        self._init_openai()
        self._init_gemini()
        self._init_google_sentiment()
        self._init_perplexity()
        
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not self.openai_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.openai_client = OpenAI(api_key=self.openai_key)
        
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        if not self.gemini_key:
            raise ValueError("Gemini API key not found in environment variables")
        genai.configure(api_key=self.gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def _init_google_sentiment(self):
        """Initialize Google Natural Language client"""
        if not self.google_sentiment_key:
            raise ValueError("Google Sentiment API key not found in environment variables")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.google_sentiment_key
        self.language_client = google_language.LanguageServiceClient()
        
    def _init_perplexity(self):
        """Initialize Perplexity client"""
        if not self.perplexity_key:
            raise ValueError("Perplexity API key not found in environment variables")
        self.perplexity_client = httpx.Client(
            base_url="https://api.perplexity.ai",
            headers={"Authorization": f"Bearer {self.perplexity_key}"}
        )
    
    async def analyze_hiring_signals(self, company: str) -> Dict[str, Any]:
        """
        Analyze hiring signals using multiple APIs for comprehensive insights.
        """
        results = {
            'company': company,
            'signals': [],
            'sentiment': None,
            'insights': None,
            'confidence': 0.0
        }
        
        try:
            # Get initial signals from Perplexity
            perplexity_data = await self._get_perplexity_signals(company)
            results['signals'].extend(perplexity_data)
            
            # Analyze sentiment
            sentiment = await self._analyze_sentiment(str(perplexity_data))
            results['sentiment'] = sentiment
            
            # Get deeper insights from GPT-4
            insights = await self._get_gpt4_insights(company, perplexity_data, sentiment)
            results['insights'] = insights
            
            # Validate with Gemini
            validation = await self._validate_with_gemini(company, insights)
            results['confidence'] = validation['confidence']
            
            return results
            
        except Exception as e:
            print(f"Error analyzing hiring signals: {str(e)}")
            return results
    
    async def _get_perplexity_signals(self, company: str) -> list:
        """Get hiring signals from Perplexity API"""
        query = f"Recent hiring signals, funding, and growth indicators for {company}"
        response = await self.perplexity_client.post(
            "/api/search",
            json={"query": query}
        )
        return response.json()
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using Google Natural Language API"""
        document = google_language.Document(
            content=text,
            type_=google_language.Document.Type.PLAIN_TEXT
        )
        sentiment = self.language_client.analyze_sentiment(document=document)
        return {
            'score': sentiment.document_sentiment.score,
            'magnitude': sentiment.document_sentiment.magnitude
        }
    
    async def _get_gpt4_insights(self, company: str, signals: list, sentiment: Dict[str, Any]) -> str:
        """Get detailed insights using GPT-4"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing hiring patterns and market signals."},
                {"role": "user", "content": f"Analyze these hiring signals and sentiment for {company}: {signals}\nSentiment: {sentiment}"}
            ]
        )
        return response.choices[0].message.content
    
    async def _validate_with_gemini(self, company: str, insights: str) -> Dict[str, float]:
        """Validate insights using Gemini"""
        response = await self.gemini_model.generate_content(
            f"Validate these hiring insights for {company}: {insights}"
        )
        return {
            'confidence': float(response.result.score) if hasattr(response.result, 'score') else 0.7
        }

@lru_cache()
def get_api_client() -> APIClient:
    """Get or create API client instance"""
    return APIClient()
