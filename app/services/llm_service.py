"""
LLM service for generating responses using various providers.
"""
import os
import time
import logging
from typing import Dict, Any, List, Optional
import openai
import anthropic
import ollama
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with various LLM providers."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        
        # Initialize OpenAI client
        if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            openai.api_key = settings.openai_api_key
        
        # Initialize Anthropic client
        if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            self.anthropic_client = None
        
        # Initialize Ollama client
        if hasattr(settings, 'ollama_base_url') and settings.ollama_base_url != "http://localhost:11434":
            # For newer versions of ollama client, we can set the host via environment variable
            os.environ["OLLAMA_HOST"] = settings.ollama_base_url
        
        # Initialize Gemini client
        if hasattr(settings, 'gemini_api_key') and settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(settings.gemini_model)
        else:
            self.gemini_model = None
        
        logger.info(f"LLM service initialized with provider: {self.provider}")
    
    def generate_response(self, query: str, context: str, 
                         additional_context: Dict[str, Any] = None) -> str:
        """Generate a response using the configured LLM provider."""
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                response = self._generate_openai_response(query, context, additional_context)
            elif self.provider == "anthropic":
                response = self._generate_anthropic_response(query, context, additional_context)
            elif self.provider == "ollama":
                response = self._generate_ollama_response(query, context, additional_context)
            elif self.provider == "gemini":
                response = self._generate_gemini_response(query, context, additional_context)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            processing_time = time.time() - start_time
            logger.info(f"Generated response in {processing_time:.2f}s with confidence 0.00")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _generate_openai_response(self, query: str, context: str,
                                 additional_context: Dict[str, Any] = None) -> str:
        """Generate response using OpenAI API."""
        try:
            system_prompt = self._build_system_prompt(additional_context)
            user_prompt = self._build_user_prompt(query, context)
            
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _generate_anthropic_response(self, query: str, context: str,
                                    additional_context: Dict[str, Any] = None) -> str:
        """Generate response using Anthropic API."""
        try:
            system_prompt = self._build_system_prompt(additional_context)
            user_prompt = self._build_user_prompt(query, context)
            
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _generate_ollama_response(self, query: str, context: str,
                                  additional_context: Dict[str, Any] = None) -> str:
        """Generate response using Ollama API."""
        try:
            system_prompt = self._build_system_prompt(additional_context)
            user_prompt = self._build_user_prompt(query, context)
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _generate_gemini_response(self, query: str, context: str,
                                 additional_context: Dict[str, Any] = None) -> str:
        """Generate response using Gemini API."""
        try:
            system_prompt = self._build_system_prompt(additional_context)
            user_prompt = self._build_user_prompt(query, context)
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.gemini_model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _generate_follow_up_questions(self, query: str, context: str,
                                     additional_context: Dict[str, Any] = None) -> List[str]:
        """Generate follow-up questions using the configured LLM provider."""
        try:
            if self.provider == "openai":
                return self._generate_openai_follow_up_questions(query, context, additional_context)
            elif self.provider == "anthropic":
                return self._generate_anthropic_follow_up_questions(query, context, additional_context)
            elif self.provider == "ollama":
                return self._generate_ollama_follow_up_questions(query, context, additional_context)
            elif self.provider == "gemini":
                return self._generate_gemini_follow_up_questions(query, context, additional_context)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def _generate_openai_follow_up_questions(self, query: str, context: str,
                                            additional_context: Dict[str, Any] = None) -> List[str]:
        """Generate follow-up questions using OpenAI."""
        try:
            prompt = f"""Based on the query "{query}" and the context provided, generate 3 relevant follow-up questions that would help clarify or expand on the topic. Return only the questions, one per line."""
            
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            questions = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
        except Exception as e:
            logger.error(f"OpenAI follow-up questions error: {e}")
            return []
    
    def _generate_anthropic_follow_up_questions(self, query: str, context: str,
                                               additional_context: Dict[str, Any] = None) -> List[str]:
        """Generate follow-up questions using Anthropic."""
        try:
            prompt = f"""Based on the query "{query}" and the context provided, generate 3 relevant follow-up questions that would help clarify or expand on the topic. Return only the questions, one per line."""
            
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            questions = response.content[0].text.strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
        except Exception as e:
            logger.error(f"Anthropic follow-up questions error: {e}")
            return []
    
    def _generate_ollama_follow_up_questions(self, query: str, context: str,
                                            additional_context: Dict[str, Any] = None) -> List[str]:
        """Generate follow-up questions using Ollama."""
        try:
            prompt = f"""Based on the query "{query}" and the context provided, generate 3 relevant follow-up questions that would help clarify or expand on the topic. Return only the questions, one per line."""
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 200
                }
            )
            
            questions = response['message']['content'].strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
        except Exception as e:
            logger.error(f"Ollama follow-up questions error: {e}")
            return []
    
    def _generate_gemini_follow_up_questions(self, query: str, context: str,
                                             additional_context: Dict[str, Any] = None) -> List[str]:
        """Generate follow-up questions using Gemini."""
        try:
            prompt = f"""Based on the query "{query}" and the context provided, generate 3 relevant follow-up questions that would help clarify or expand on the topic. Return only the questions, one per line."""
            
            response = self.gemini_model.generate_content(prompt)
            questions = response.text.strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
        except Exception as e:
            logger.error(f"Gemini follow-up questions error: {e}")
            return []
    
    def _build_system_prompt(self, additional_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for the LLM."""
        base_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Always base your answers on the given context and cite specific parts when possible. 
        If the context doesn't contain enough information to answer the question, say so clearly.
        Be concise, accurate, and helpful."""
        
        if additional_context:
            context_str = "\n".join([f"{k}: {v}" for k, v in additional_context.items()])
            base_prompt += f"\n\nAdditional context:\n{context_str}"
        
        return base_prompt
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt for the LLM."""
        return f"""Context: {context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider configuration."""
        info = {
            "current_provider": self.provider,
            "current_model": self.model,
            "openai_configured": bool(settings.openai_api_key),
            "anthropic_configured": bool(settings.anthropic_api_key),
            "ollama_configured": bool(hasattr(settings, 'ollama_base_url')),
            "ollama_base_url": getattr(settings, 'ollama_base_url', None),
            "gemini_configured": bool(hasattr(settings, 'gemini_api_key') and settings.gemini_api_key),
            "gemini_model": getattr(settings, 'gemini_model', None)
        }
        return info
    
    def update_provider(self, provider: str, model: str = None):
        """Update the LLM provider and model."""
        valid_providers = ["openai", "anthropic", "ollama", "gemini"]
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider. Must be one of: {valid_providers}")
        
        self.provider = provider
        if model:
            self.model = model
        
        logger.info(f"Updated LLM provider to {provider} with model {self.model}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers based on configuration."""
        providers = []
        
        if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            providers.append("openai")
        
        if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
            providers.append("anthropic")
        
        if hasattr(settings, 'ollama_base_url'):
            providers.append("ollama")
        
        if hasattr(settings, 'gemini_api_key') and settings.gemini_api_key:
            providers.append("gemini")
        
        return providers
