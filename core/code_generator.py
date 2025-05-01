"""
Code Generator module for Java Peer Review Training System.

This module provides the CodeGenerator class which dynamically generates
Java code snippets based on the selected difficulty level and code length,
eliminating the reliance on predefined templates.
"""

import random
import logging
from langchain_core.language_models import BaseLanguageModel
from utils.code_utils import create_code_generation_prompt
from utils.llm_logger import LLMInteractionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    Generates Java code snippets dynamically without relying on predefined templates.
    This class creates realistic Java code based on specified complexity and length.
    """
    def __init__(self, llm: BaseLanguageModel = None, llm_logger: LLMInteractionLogger = None):
        """
        Initialize the CodeGenerator with an optional language model.
        
        Args:
            llm: Language model to use for code generation
            llm_logger: Logger for tracking LLM interactions
        """
        self.llm = llm
        self.llm_logger = llm_logger or LLMInteractionLogger()
        
        # Define complexity profiles for different code lengths
        self.complexity_profiles = {
            "short": {
                "class_count": 1,
                "method_count_range": (1, 2),  # Reduced to 1-2 methods for beginners
                "field_count_range": (1, 3),   # Fewer fields
                "imports_count_range": (0, 1), # Minimal imports
                "nested_class_prob": 0.0,      # No nested classes for beginners
                "interface_prob": 0.0          # No interfaces for beginners
            },
            "medium": {
                "class_count": 1,
                "method_count_range": (3, 5),  # Reduced from 3-6 to 3-5
                "field_count_range": (2, 5),   # Reduced field count
                "imports_count_range": (0, 3), # Fewer imports
                "nested_class_prob": 0.1,      # Reduced probability of nested classes
                "interface_prob": 0.1          # Reduced probability of interfaces
            },
            "long": {
                "class_count": 1,              # Changed from 2 to 1 (with possibility of 2)
                "method_count_range": (4, 8),  # Reduced from 5-10 to 4-8
                "field_count_range": (3, 6),   # Reduced from 4-8 to 3-6
                "imports_count_range": (1, 4), # Reduced from 2-6 to 1-4
                "nested_class_prob": 0.3,      # Reduced from 0.5 to 0.3
                "interface_prob": 0.2          # Reduced from 0.4 to 0.2
            }
        }
        
        # Common Java domains to make code more realistic
        self.domains = [
            "user_management", "file_processing", "data_validation", 
            "calculation", "inventory_system", "notification_service",
            "logging", "banking", "e-commerce", "student_management"
        ]
     
    def _generate_with_llm(self, code_length: str, difficulty_level: str, domain: str = None, 
                       selected_errors=None) -> str:
        """
        Generate Java code using the language model.
        Handles both Ollama and Groq API responses.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Optional domain for the code context
            selected_errors: Optional list of errors to include
            
        Returns:
            Generated Java code as a string or AIMessage object
        """
        # Select a domain if not provided
        if not domain:
            domain = random.choice(self.domains)
        
        # Create a detailed prompt for the LLM using shared utility
        prompt = create_code_generation_prompt(
            code_length=code_length,
            difficulty_level=difficulty_level,
            selected_errors=selected_errors or [],  # No errors for clean code
            domain=domain,
            include_error_annotations=False if selected_errors is None else True
        )
            
        try:
            # Metadata for logging
            metadata = {
                "code_length": code_length,
                "difficulty_level": difficulty_level,
                "domain": domain,
                "selected_errors": selected_errors or []
            }
            
            # Add provider info to metadata if available
            if hasattr(self.llm, 'provider'):
                metadata["provider"] = self.llm.provider
                logger.info(f"Generating Java code with provider: {self.llm.provider}")
            elif hasattr(self.llm, 'model_name') and 'groq' in type(self.llm).__name__.lower():
                metadata["provider"] = "groq"
                logger.info(f"Generating Java code with Groq model: {self.llm.model_name}")
            else:
                logger.info(f"Generating Java code with LLM: {code_length} length, {difficulty_level} difficulty, {domain} domain")
            
            # Generate the code using the LLM
            response = self.llm.invoke(prompt)
            
            # Log the response type
            logger.info(f"LLM response type: {type(response).__name__}")
            
            # Log to the LLM logger
            self.llm_logger.log_code_generation(prompt, response, metadata)
            
            # Return the response (can be string or AIMessage depending on provider)
            return response
            
        except Exception as e:
            logger.error(f"Error generating code with LLM: {str(e)}")          
            return """
    """
           
    