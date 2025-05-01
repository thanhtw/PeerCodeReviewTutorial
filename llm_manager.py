"""
LLM Manager module for Java Peer Review Training System.

This module provides the LLMManager class for handling model initialization,
configuration, and management of LLM providers including Ollama and Groq.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv 
import inspect

from langchain_community.llms.ollama import Ollama

# Add Groq integration
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage
GROQ_AVAILABLE = True

from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMManager:
    """
    LLM Manager for handling model initialization, configuration and management.
    Supports multiple LLM providers including Ollama and Groq.
    """
    
    def __init__(self):
        """Initialize the LLM Manager with environment variables."""
        load_dotenv(override=True)
        
        # Provider settings - default to Ollama
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        # Ollama settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("DEFAULT_MODEL", "llama3:1b")
        
        # Groq settings
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_api_base = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        self.groq_default_model = os.getenv("GROQ_DEFAULT_MODEL", "llama3-8b-8192")
        
        # Available Groq models
        self.groq_available_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        # Force GPU usage by default if available (Ollama only)
        self.force_gpu = os.getenv("ENABLE_GPU", "true").lower() == "true"

        # Get the GPU_LAYERS value
        gpu_layers_value = os.getenv("GPU_LAYERS", "-1")

        # Extract just the number part in case there are comments in the env var
        if gpu_layers_value and "#" in gpu_layers_value:
            gpu_layers_value = gpu_layers_value.split("#")[0].strip()
            
        self.gpu_layers = int(gpu_layers_value)
        # Track initialized models

        self.initialized_models = {}
        
        # Cache GPU info (Ollama only)
        self._gpu_info = None

    def refresh_gpu_info(self):
        """Refresh GPU information with extended details."""
        self._gpu_info = self.check_gpu_availability(extended=True)
        return self._gpu_info
       
    def set_provider(self, provider: str, api_key: str = None) -> bool:
        """
        Set the LLM provider to use and persist the selection.
        
        Args:
            provider: Provider name ('ollama' or 'groq')
            api_key: API key for Groq (required if provider is 'groq')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if provider.lower() not in ["ollama", "groq"]:
            logger.error(f"Unsupported provider: {provider}")
            return False
            
        # Set the provider in instance and persist to environment
        self.provider = provider.lower()
        os.environ["LLM_PROVIDER"] = provider.lower()
        logger.info(f"Provider set to: {self.provider}")
        
        # Clear initialized models to force reinitialization
        self.initialized_models = {}
        
        # Handle provider-specific setup
        if self.provider == "groq":
            if not GROQ_AVAILABLE:
                logger.error("Groq integration is not available. Please install langchain-groq package.")
                return False
                
            # Validate and set API key
            if not api_key and not self.groq_api_key:
                logger.error("API key is required for Groq provider")
                return False
                
            if api_key:
                self.groq_api_key = api_key
                os.environ["GROQ_API_KEY"] = api_key
            
            # Test the API key
            if not self.check_groq_connection()[0]:
                logger.error("Failed to connect to Groq API. Please check your API key.")
                return False
        
        elif self.provider == "ollama":            
            # Test connection to Ollama
            if not self.check_ollama_connection()[0]:
                logger.warning("Failed to connect to Ollama. Please ensure Ollama is running.")
                return False
        
        # Log successful provider change
        logger.info(f"Successfully switched to {self.provider} provider")
        return True    
    
    def _format_size(self, size_in_bytes: int) -> str:
        """Format size in human-readable format."""
        if not isinstance(size_in_bytes, (int, float)):
            return "Unknown"
            
        size = float(size_in_bytes)
        
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
            
        return f"{size:.2f} TB"

    #check connection        
    def check_ollama_connection(self) -> Tuple[bool, str]:
        """
        Check if Ollama service is running and accessible.
        
        Returns:
            Tuple[bool, str]: (is_connected, message)
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True, "Connected to Ollama successfully"
            else:
                return False, f"Connected to Ollama but received status code {response.status_code}"
        except requests.ConnectionError:
            return False, f"Failed to connect to Ollama at {self.ollama_base_url}"
        except Exception as e:
            return False, f"Error checking Ollama connection: {str(e)}"
    
    def check_groq_connection(self) -> Tuple[bool, str]:
        """
        Check if Groq API is accessible with the current API key.
        
        Returns:
            Tuple[bool, str]: (is_connected, message)
        """
        if not self.groq_api_key:
            return False, "No Groq API key provided"
            
        if not GROQ_AVAILABLE:
            return False, "Groq integration is not available. Please install langchain-groq package."
            
        try:
            # We'll use a minimal API call to test the connection
            # Using the ChatGroq class from langchain_groq
            chat = ChatGroq(
                api_key=self.groq_api_key,
                model_name="llama3-8b-8192"  # Use the smallest model for testing
            )
            
            # Make a minimal API call
            response = chat.invoke([HumanMessage(content="test")])
            
            # If we get here, the connection is successful
            return True, "Connected to Groq API successfully"
            
        except Exception as e:
            error_message = str(e)
            if "auth" in error_message.lower() or "api key" in error_message.lower():
                return False, "Invalid Groq API key"
            else:
                return False, f"Error connecting to Groq API: {error_message}"

    #initial_model
    def initialize_model(self, model_name: str, model_params: Dict[str, Any] = None) -> Optional[BaseLanguageModel]:
        """
        Initialize a model with the current provider.
        
        Args:
            model_name (str): Name of the model to initialize
            model_params (Dict[str, Any], optional): Model parameters
            
        Returns:
            Optional[BaseLanguageModel]: Initialized LLM or None if initialization fails
        """
        # Create a unique key for caching based on model name, provider, and params
        # cache_key = f"{self.provider}_{model_name}"
        
        # if cache_key in self.initialized_models:
        #     logger.info(f"Using cached model: {cache_key}")
        #     return self.initialized_models[cache_key]
        
        # Use the appropriate initialization method based on the provider
        if self.provider == "ollama":
            model = self._initialize_ollama_model(model_name, model_params)
        elif self.provider == "groq":
            model = self._initialize_groq_model(model_name, model_params)
        else:
            logger.error(f"Unsupported provider: {self.provider}")
            return None
            
        # Cache the model if initialization was successful
        # if model:
        #     self.initialized_models[cache_key] = model
            
        return model
    
    def _initialize_groq_model(self, model_name: str, model_params: Dict[str, Any] = None) -> Optional[BaseLanguageModel]:
        """
        Initialize a Groq model.
        
        Args:
            model_name: Name of the model to initialize
            model_params: Model parameters
            
        Returns:
            Initialized LLM or None if initialization fails
        """
        if not GROQ_AVAILABLE:
            logger.error("Groq integration is not available. Please install langchain-groq package.")
            return None
            
        if not self.groq_api_key:
            logger.error("No Groq API key provided")
            return None
            
        # Apply default model parameters if none provided
        if model_params is None:
            model_params = self._get_groq_default_params(model_name)
            
        try:
            # Initialize the Groq model
            temperature = model_params.get("temperature", 0.7)
            
            # Use the ChatGroq class from langchain_groq
            llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name=model_name,
                temperature=temperature,
                verbose=True
            )
            
            # Test with a simple message to ensure the model works
            try:
                _ = llm.invoke([HumanMessage(content="test")])
                logger.info(f"Successfully initialized Groq model: {model_name}")
                return llm
            except Exception as e:
                logger.error(f"Error testing Groq model {model_name}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error initializing Groq model {model_name}: {str(e)}")
            return None
    
    def _initialize_ollama_model(self, model_name: str, model_params: Dict[str, Any] = None) -> Optional[BaseLanguageModel]:
        """
        Initialize an Ollama model with improved GPU support.
        
        Args:
            model_name: Name of the model to initialize
            model_params: Model parameters
            
        Returns:
            Initialized LLM or None if initialization fails
        """
        # Apply default model parameters if none provided
        if model_params is None:
            model_params = {}
        
        # Ensure GPU is always enabled if available and forced
        if self.force_gpu:
            # Get GPU info with comprehensive detection
            gpu_info = self.check_gpu_availability(extended=True)
            
            if gpu_info and gpu_info.get("has_gpu", False):
                # Always set GPU parameters for better performance
                gpu_layers = self.gpu_layers if self.gpu_layers > 0 else -1  # -1 means use all layers
                model_params["n_gpu_layers"] = gpu_layers
                model_params["f16_kv"] = True  # Use half-precision for key/value cache to save GPU memory
                
                logger.info(f"Enabling GPU acceleration for Ollama model {model_name} with {gpu_layers} layers")
        
        # Initialize Ollama model
        try:
            # Check if model is available
            if not self._check_ollama_model_availability(model_name):
                logger.warning(f"Model {model_name} not found. Attempting to pull...")
                if self.download_ollama_model(model_name):
                    logger.info(f"Successfully pulled model {model_name}")
                else:
                    logger.error(f"Failed to pull model {model_name}")
                    return None
            
            # Extract supported parameters
            temperature = model_params.get("temperature", 0.7)
            ollama_params = {}
            
            # Only include supported parameters
            supported_params = [
                "temperature", 
                "model", 
                "base_url", 
                "keep_alive",
                "num_ctx",
                "repeat_penalty",
                "top_k",
                "top_p"
            ]
            
            for key, value in model_params.items():
                if key in supported_params:
                    ollama_params[key] = value
            
            # Handle GPU layers separately as a model kwarg
            if "n_gpu_layers" in model_params:
                # Try to use num_gpu parameter via model_kwargs if available
                gpu_layers = model_params["n_gpu_layers"]
                
                if hasattr(Ollama, "model_kwargs") or "model_kwargs" in inspect.signature(Ollama.__init__).parameters:
                    ollama_params["model_kwargs"] = {"num_gpu": gpu_layers}
                else:
                    # Older versions - try direct parameter
                    logger.info(f"Using older Ollama client style for GPU layers: {gpu_layers}")
                    # Only include if positive to avoid validation errors
                    if gpu_layers > 0:
                        ollama_params["num_gpu"] = gpu_layers
            
            # Log the parameters being used
            logger.info(f"Initializing model {model_name} with params: {ollama_params}")
            
            # Create the Ollama model 
            try:
                llm = Ollama(
                    base_url=self.ollama_base_url,
                    model=model_name,
                    temperature=temperature,
                    **ollama_params
                )
            except TypeError as e:
                # If the above fails due to unexpected parameters, try with minimal params
                logger.warning(f"Error with full params: {str(e)}, trying minimal params")
                llm = Ollama(
                    base_url=self.ollama_base_url,
                    model=model_name,
                    temperature=temperature
                )
            
            # Test the model with a simple query
            try:
                _ = llm.invoke("hello")
                
                # Log GPU status
                if self.force_gpu:
                    gpu_info = self.check_gpu_availability()
                    if gpu_info and gpu_info.get("has_gpu", False):
                        logger.info(f"Successfully initialized {model_name} with GPU acceleration")
                    else:
                        logger.info(f"Successfully initialized {model_name} (CPU only)")
                
                return llm
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {str(e)}")
                return None
                    
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            return None
    
    def initialize_model_from_env(self, model_key: str, temperature_key: str) -> Optional[BaseLanguageModel]:
        """
        Initialize a model using environment variables with enhanced GPU awareness.
        
        Args:
            model_key (str): Environment variable key for model name
            temperature_key (str): Environment variable key for temperature
            
        Returns:
            Optional[BaseLanguageModel]: Initialized LLM or None if initialization fails
        """
        # Ensure we're using the correct provider
        current_provider = self.provider.lower()
        logger.info(f"Initializing model from env with provider: {current_provider}")
        
        # Get model name based on provider
        if current_provider == "ollama":
            model_name = os.getenv(model_key, self.default_model)
            logger.info(f"Using Ollama model: {model_name}")
        elif current_provider == "groq":
            # First check for Groq-specific model environment variable
            groq_model_key = f"GROQ_{model_key}"
            model_name = os.getenv(groq_model_key, self.groq_default_model)
            
            # Map environment variable names to Groq model names if needed
            if model_name == "llama3:8b":
                model_name = "llama3-8b-8192"
            elif model_name == "llama3:70b":
                model_name = "llama3-70b-8192"
            
            logger.info(f"Using Groq model: {model_name}")
        else:
            logger.error(f"Unsupported provider: {current_provider}")
            return None
        
        # Get temperature
        temperature = float(os.getenv(temperature_key, "0.7"))       
       
        
        # Set up basic model parameters
        model_params = {
            "temperature": temperature
        }
        
        # Aggressively ensure GPU is enabled for Ollama
        if current_provider == "ollama":
            # Force enable GPU by default
            if self.force_gpu:
                # Check GPU availability with comprehensive detection
                gpu_info = self.check_gpu_availability(extended=True)
                if gpu_info.get("has_gpu", False):
                    # Always set GPU parameters for Ollama models
                    gpu_name = gpu_info.get("gpu_name", "GPU")
                    logger.info(f"GPU acceleration enabled: {gpu_name}")
                    
                    # Set GPU-specific parameters
                    model_params["n_gpu_layers"] = self.gpu_layers
                    model_params["f16_kv"] = True  # Use half-precision for key/value cache
                    model_params["logits_all"] = False  # Don't compute logits for all tokens (faster)
                    
                    # Adjust GPU layers based on available memory if possible
                    if "memory_total" in gpu_info and isinstance(gpu_info["memory_total"], (int, float)):
                        memory_gb = gpu_info["memory_total"] / (1024 * 1024 * 1024)
                        if memory_gb < 4:
                            # Very limited GPU memory - be more conservative
                            model_params["n_gpu_layers"] = min(24, self.gpu_layers if self.gpu_layers > 0 else 24)
                        elif memory_gb < 8:
                            # Limited GPU memory
                            model_params["n_gpu_layers"] = min(32, self.gpu_layers if self.gpu_layers > 0 else 32)
                else:
                    logger.warning("GPU not available, using CPU for inference")
        
        # Initialize the model with the provider-specific settings
        logger.info(f"Initializing model {model_name} with params: {model_params}")
        return self.initialize_model(model_name, model_params)
    
    def _get_groq_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a Groq model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Default parameters for the model
        """
        # Basic defaults for Groq
        params = {
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        # Adjust based on model name and role
        if "generative" in model_name or "llama3" in model_name:
            params["temperature"] = 0.8  # Slightly higher creativity for generative tasks
        elif "review" in model_name or "mixtral" in model_name:
            params["temperature"] = 0.3  # Lower temperature for review tasks
        elif "summary" in model_name:
            params["temperature"] = 0.4  # Moderate temperature for summary tasks
        elif "compare" in model_name:
            params["temperature"] = 0.5  # Balanced temperature for comparison tasks
        
        return params
    
   