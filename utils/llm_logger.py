"""
Enhanced LLM Interaction Logger for Java Peer Review Training System.

This module provides utilities for logging LLM interactions, ensuring proper
extraction and formatting of responses with improved readability.
"""

import os
import json
import datetime
import logging
import re
from typing import List, Any, Dict, Optional
from pathlib import Path

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
    
    def __init__(self, log_dir: str = "llm_logs"):
        """
        Initialize the LLM interaction logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        self.logs = []
        
        # Create log directory if it doesn't exist
        self.ensure_log_directory()
        
        # Track attempt counts for different interaction types
        self._attempt_counts = {}
    
    def ensure_log_directory(self):
        """Create the log directory structure if it doesn't exist."""
        log_path = Path(self.log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created log directory: {self.log_dir}")
            
        # Create subdirectories for different log types
        log_types = [
            "code_generation", 
            "code_regeneration", 
            "code_evaluation", 
            "review_analysis", 
            "summary_generation"
        ]
        
        for log_type in log_types:
            type_path = log_path / log_type
            if not type_path.exists():
                type_path.mkdir(parents=True, exist_ok=True)
    
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
    
    def _format_for_readability(self, text: str) -> str:
        """
        Format text for better readability in log files.
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text with proper line breaks and indentation
        """
        if not text:
            return ""
            
        # Ensure proper line breaks
        text = text.replace("\\n", "\n")
        
        # Pretty format JSON in the text if present
        def format_json_match(match):
            try:
                json_str = match.group(1)
                parsed = json.loads(json_str)
                return f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except:
                return match.group(0)
                
        # Find and format JSON blocks
        text = re.sub(r'```json\s*({.*?})\s*```', format_json_match, text, flags=re.DOTALL)
        
        # Ensure code blocks have proper formatting
        text = re.sub(r'```(\w+)?\s*\n?', r'```\1\n', text)
        text = re.sub(r'\n?\s*```', r'\n```', text)
        
        return text
    
    def get_attempt_count(self, interaction_type: str) -> int:
        """
        Get the number of attempts for a given interaction type.
        
        Args:
            interaction_type: Type of interaction
            
        Returns:
            Number of attempts
        """
        return self._attempt_counts.get(interaction_type, 0)
    
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
        
        # Format for readability
        formatted_prompt = self._format_for_readability(prompt)
        formatted_response = self._format_for_readability(processed_response)
        
        # Create a log entry
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": interaction_type,
            "prompt": formatted_prompt,
            "response": formatted_response,
            "metadata": metadata or {}
        }
        
        # Add to in-memory logs
        self.logs.append(log_entry)
        
        # Update attempt count
        self._attempt_counts[interaction_type] = self._attempt_counts.get(interaction_type, 0) + 1
        
        # Create log directory for this interaction type if it doesn't exist
        type_dir = os.path.join(self.log_dir, interaction_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # Format timestamp for filename
        file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(type_dir, f"{file_timestamp}.json")
        
        try:
            # Write formatted log entry
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
                
            # Also write a more readable text version for direct inspection
            text_file = os.path.join(type_dir, f"{file_timestamp}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"TYPE: {interaction_type}\n")
                f.write(f"METADATA: {json.dumps(metadata or {}, indent=2)}\n\n")
                f.write("=== PROMPT ===\n\n")
                f.write(formatted_prompt)
                f.write("\n\n=== RESPONSE ===\n\n")
                f.write(formatted_response)
                
            logger.debug(f"Logged {interaction_type} interaction to {log_file} and {text_file}")
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
        self.log_interaction("code_regeneration", prompt, response, metadata)
    
    def log_regeneration_prompt(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a regeneration prompt (before it's sent to the LLM).
        
        Args:
            prompt: The regeneration prompt that will be sent to the LLM
            metadata: Optional metadata about the prompt creation
        """
        self.log_interaction("regeneration_prompt", prompt, "N/A - Prompt Only", metadata)

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
        
    def export_logs(self, export_dir: str = None) -> str:
        """
        Export all logs to a ZIP file.
        
        Args:
            export_dir: Directory to place the export file
            
        Returns:
            Path to the exported ZIP file
        """
        import zipfile
        import tempfile
        import shutil
        
        # Use the specified directory or a temporary one
        if export_dir is None:
            export_dir = tempfile.mkdtemp()
        else:
            os.makedirs(export_dir, exist_ok=True)
        
        # Create the export filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = os.path.join(export_dir, f"llm_logs_{timestamp}.zip")
        
        # Create the ZIP file
        with zipfile.ZipFile(export_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all log files
            for root, _, files in os.walk(self.log_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(self.log_dir))
                    zipf.write(file_path, arcname)
        
        logger.info(f"Logs exported to {export_filename}")
        return export_filename