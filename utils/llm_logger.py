"""
LLM Interaction Logger for Java Peer Review Training System.

This module provides utilities for logging LLM interactions, ensuring proper
extraction and formatting of responses.
"""

import os
import json
import datetime
import logging
from typing import List, Any, Dict, Optional

from utils.code_utils import process_llm_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMInteractionLogger:
    """
    Logger for LLM interactions to track prompts, responses, and metadata.
    
    This class handles logging of all interactions with the LLM,
    including prompts, responses, and relevant metadata.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the LLM interaction logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        self.logs = []
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def _ensure_string_response(self, response: Any) -> str:
        """
        Ensure the response is a string, extracting content if needed.
        
        Args:
            response: Response object from LLM
            
        Returns:
            String content of the response
        """
        # Use the utility function from code_utils for consistent processing
        return process_llm_response(response)
    
    def log_interaction(self, interaction_type: str, prompt: str, response: Any, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an LLM interaction with enhanced response processing.
        
        Args:
            interaction_type: Type of interaction (e.g., generation, evaluation)
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            metadata: Optional metadata about the interaction
        """
        # Process the response to ensure it's a string
        processed_response = self._ensure_string_response(response)
        
        # Create a log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": interaction_type,
            "prompt": prompt,
            "response": processed_response,
            "metadata": metadata or {}
        }
        
        # Add to in-memory logs
        self.logs.append(log_entry)
        
        # Create log file with timestamp and type
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{timestamp}_{interaction_type}.json")
        
        try:
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing log to file: {str(e)}")
    
    def log_code_generation(self, prompt: str, response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a code generation interaction with enhanced response processing.
        
        Args:
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            metadata: Optional metadata about the interaction
        """
        self.log_interaction("code_generation", prompt, response, metadata)
    
    def log_code_regeneration(self, prompt: str, response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a code regeneration interaction with enhanced response processing.
        
        Args:
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            metadata: Optional metadata about the interaction
        """
        # IMPORTANT: Don't store logs in "regeneration_prompt" folder
        self.log_interaction("code_regeneration", prompt, response, metadata)
    
    def log_code_evaluation(self, prompt: str, response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a code evaluation interaction with enhanced response processing.
        
        Args:
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            metadata: Optional metadata about the interaction
        """
        self.log_interaction("code_evaluation", prompt, response, metadata)
    
    def log_review_analysis(self, prompt: str, response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a review analysis interaction with enhanced response processing.
        
        Args:
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            metadata: Optional metadata about the interaction
        """
        self.log_interaction("review_analysis", prompt, response, metadata)
    
    def log_summary_generation(self, prompt: str, response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a summary generation interaction with enhanced response processing.
        
        Args:
            prompt: Prompt sent to the LLM
            response: Response from the LLM
            metadata: Optional metadata about the interaction
        """
        self.log_interaction("summary_generation", prompt, response, metadata)
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent logs for display in the UI.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of recent log entries
        """
        return self.logs[-limit:] if len(self.logs) > limit else self.logs
    
    def clear_logs(self) -> None:
        """Clear in-memory logs."""
        self.logs = []