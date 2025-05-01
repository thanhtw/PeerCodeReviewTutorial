"""
Unified Code Evaluation Agent for Java Peer Review Training System.

This module provides the CodeEvaluationAgent class which evaluates 
generated Java code to ensure it contains the required errors.
Incorporates enhanced evaluation methods for more accurate analysis.
"""

import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

from utils.llm_logger import LLMInteractionLogger
from utils.code_utils import create_evaluation_prompt, create_regeneration_prompt, process_llm_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeEvaluationAgent:
    """
    Agent for evaluating generated Java code to ensure it meets error requirements.
    
    This agent provides detailed feedback on how well the generated code
    implements the required errors, and suggests improvements for the
    code generator. Can use an LLM for more accurate evaluation.
    """
    
    def __init__(self, llm: BaseLanguageModel = None, llm_logger = None):
        """
        Initialize the CodeEvaluationAgent.
        
        Args:
            llm: Language model for evaluation
            llm_logger: Logger for tracking LLM interactions
        """
        self.llm = llm
        self.llm_logger = llm_logger
    
    def evaluate_code(self, code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate Java code to check for requested errors.
        
        Args:
            code: The Java code to evaluate
            requested_errors: List of errors that should be included in the code
            
        Returns:
            Evaluation results with found and missing errors
        """
        # Default result if no evaluation can be performed
        default_result = {
            "found_errors": [],
            "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                            for error in requested_errors],
            "valid": False,
            "feedback": f"Could not evaluate code. Please ensure the code contains all {len(requested_errors)} requested errors."
        }
        
        # Check if LLM is available for evaluation
        if not self.llm:
            logger.warning("No LLM available for code evaluation")
            return default_result
        
        # Create evaluation prompt
        prompt = create_evaluation_prompt(code, requested_errors)
        
        try:
            # Generate the evaluation using the LLM
            logger.info("Sending code to LLM for evaluation")
            response = self.llm.invoke(prompt)
            # Process response to ensure it's properly formatted
            processed_response = process_llm_response(response)
            
            # Log the evaluation
            if self.llm_logger:
                metadata = {
                    "code_length": len(code.splitlines()),
                    "requested_errors_count": len(requested_errors)
                }
                self.llm_logger.log_code_evaluation(prompt, processed_response, metadata)
            
            # Extract JSON from the response
            evaluation_result = self._extract_json_from_response(processed_response)
            
            # IMPORTANT: Check if extraction failed and return default result if it did
            if not evaluation_result or not isinstance(evaluation_result, dict):
                logger.warning("Failed to extract JSON from evaluation response or result is not a dictionary")
                return default_result
            
            # Process the evaluation result
            processed_result = self._process_evaluation_result(evaluation_result, requested_errors)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error evaluating code: {str(e)}")
            return default_result
    
    def generate_improved_prompt(self, code: str, requested_errors: List[Dict[str, Any]], 
                          evaluation: Dict[str, Any]) -> str:
        """
        Generate an improved prompt for the code generator based on evaluation results.
        
        Args:
            code: The previously generated code
            requested_errors: List of errors that should be implemented
            evaluation: Evaluation results from evaluate_code method
            
        Returns:
            Improved prompt string for the code generator
        """       
        
        # Determine domain from existing code
        domain = self._infer_domain_from_code(code)
        
        # Extract missing and found errors
        missing_errors = []
        found_errors = []
        
        # Process missing errors - handle both string and dictionary formats
        if "missing_errors" in evaluation:
            for error in evaluation["missing_errors"]:
                if isinstance(error, dict):
                    error_type = error.get("error_type", "").upper()
                    error_name = error.get("error_name", "")
                    missing_errors.append(f"{error_type} - {error_name}")
                elif isinstance(error, str):
                    missing_errors.append(error)
        
        # Process found errors - handle both string and dictionary formats
        if "found_errors" in evaluation:
            for error in evaluation["found_errors"]:
                if isinstance(error, dict):
                    error_type = error.get("error_type", "").upper()
                    error_name = error.get("error_name", "")
                    found_errors.append(f"{error_type} - {error_name}")
                elif isinstance(error, str):
                    found_errors.append(error)
        
        # Use the optimized prompt function

        
        prompt = create_regeneration_prompt(
            code=code,
            domain=domain,
            missing_errors=missing_errors,
            found_errors=found_errors,
            requested_errors=requested_errors
        )
        
        # Log the regeneration prompt
        metadata = {
            "requested_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" for error in requested_errors],
            "missing_errors": missing_errors,
            "found_errors": found_errors,
            "domain": domain,
            "attempt": self.llm_logger.get_attempt_count("code_generation") + 1
        }
        
        self.llm_logger.log_interaction("regeneration_prompt", prompt, "N/A - Prompt Only", metadata)
        
        return prompt

    def _infer_domain_from_code(self, code: str) -> str:
        """
        Infer the domain of the code based on class and variable names.
        
        Args:
            code: The Java code
            
        Returns:
            Inferred domain string
        """
        code_lower = code.lower()
        
        # Check for common domains
        domains = {
            "student_management": ["student", "course", "enroll", "grade", "academic"],
            "file_processing": ["file", "read", "write", "path", "directory"],
            "data_validation": ["validate", "input", "check", "valid", "sanitize"],
            "calculation": ["calculate", "compute", "math", "formula", "result"],
            "inventory_system": ["inventory", "product", "stock", "item", "quantity"],
            "notification_service": ["notify", "message", "alert", "notification", "send"],
            "banking": ["account", "bank", "transaction", "balance", "deposit"],
            "e-commerce": ["cart", "product", "order", "payment", "customer"]
        }
        
        # Count domain-related terms
        domain_scores = {}
        for domain, terms in domains.items():
            score = sum(code_lower.count(term) for term in terms)
            domain_scores[domain] = score
        
        # Return the highest scoring domain, or a default
        if domain_scores:
            max_domain = max(domain_scores.items(), key=lambda x: x[1])
            if max_domain[1] > 0:
                return max_domain[0]
        
        return "general_application"  # Default domain
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON data from LLM response with improved handling for Groq responses.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted JSON data or None if extraction fails
        """
        # Check if response is None or empty
        if not response:
            return None
        
        # Ensure response is a string
        if not isinstance(response, str):
            try:
                response = str(response)
            except:
                logger.error("Could not convert response to string")
                return None
        
        # Log first part of response for debugging
        logger.debug(f"Extracting JSON from response: {response[:200]}...")
        
        # First try direct JSON parsing if the response looks like JSON
        if response.strip().startswith('{') and response.strip().endswith('}'):
            try:
                # Clean the response to fix common JSON issues
                json_str = response.strip()
                # Fix trailing commas which are invalid in JSON
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # Try to parse as JSON directly
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If direct parsing fails, continue with regex extraction
                pass
        
        # Try to find JSON block with various patterns
        patterns = [
            r'```json\s*([\s\S]*?)```',  # JSON in code block
            r'```\s*({[\s\S]*?})\s*```',  # Any JSON in code block
            r'({[\s\S]*?"found_errors"[\s\S]*?})',  # JSON with found_errors field
            r'({[\s\S]*?"valid"[\s\S]*?})',  # JSON with valid field
            r'({[\s\S]*?"missing_errors"[\s\S]*?})',  # JSON with missing_errors field
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Clean the match to fix common JSON issues
                    json_str = match.strip()
                    # Fix trailing commas which are invalid in JSON
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    # Try to parse as JSON
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # If regex extraction fails, try to find JSON-like structure with looser matching
        try:
            opening_bracket = response.find('{')
            closing_bracket = response.rfind('}')
            
            if opening_bracket != -1 and closing_bracket != -1 and opening_bracket < closing_bracket:
                json_str = response[opening_bracket:closing_bracket + 1]
                # Fix trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # Try to parse as JSON
                return json.loads(json_str)
        except:
            pass
        
        # For Groq responses, if all extraction methods fail, try a more aggressive approach
        # to build a structured result manually
        missing_errors = []
        found_errors = []
        
        # Try to extract found_errors section
        found_match = re.search(r'found_errors:?\s*\n(.*?)(?:missing_errors|\n\n)', response, re.DOTALL)
        if found_match:
            found_section = found_match.group(1)
            # Extract individual errors
            for line in found_section.splitlines():
                if line.strip() and ":" in line:
                    found_errors.append(line.strip())
        
        # Try to extract missing_errors section
        missing_match = re.search(r'missing_errors:?\s*\n(.*?)(?:\n\n|$)', response, re.DOTALL)
        if missing_match:
            missing_section = missing_match.group(1)
            # Extract individual errors
            for line in missing_section.splitlines():
                if line.strip() and ":" in line:
                    missing_errors.append(line.strip())
        
        # If we extracted at least some structured data, return a constructed result
        if found_errors or missing_errors:
            logger.info(f"Using manually extracted errors: {len(found_errors)} found, {len(missing_errors)} missing")
            return {
                "found_errors": found_errors,
                "missing_errors": missing_errors,
                "valid": len(missing_errors) == 0,
                "feedback": f"Found {len(found_errors)} errors, {len(missing_errors)} missing."
            }
        
        # If all extraction methods fail, return a default result structure
        logger.warning("Could not extract JSON from response, returning default structure")
        # Include all requested errors as missing to force regeneration
        return {
            "found_errors": [],
            "missing_errors": ["EXTRACTION_FAILED"],  # This will force regeneration
            "valid": False,
            "feedback": "Could not extract proper analysis from model response."
        }
    
    def _process_evaluation_result(self, result: Dict[str, Any], 
                        requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and enhance the evaluation result with improved type safety.
        
        Args:
            result: Raw evaluation result from LLM
            requested_errors: List of requested errors
            
        Returns:
            Processed evaluation result
        """
        # Handle None result
        if result is None:
            logger.warning("Received None result in _process_evaluation_result")
            result = {
                "found_errors": [],
                "missing_errors": [],
                "valid": False,
                "feedback": "Failed to process evaluation result"
            }
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            logger.error(f"Expected dict for result, got {type(result)}")
            result = {
                "found_errors": [],
                "missing_errors": [],
                "valid": False,
                "feedback": f"Invalid evaluation result type: {type(result)}"
            }
        
        # Ensure all expected fields exist with proper defaults
        if "found_errors" not in result:
            result["found_errors"] = []
        if "missing_errors" not in result:
            result["missing_errors"] = []
        if "extra_errors" not in result:
            result["extra_errors"] = []
        
        # Ensure found_errors is a list
        if not isinstance(result["found_errors"], list):
            logger.warning(f"found_errors is not a list, got {type(result['found_errors'])}")
            result["found_errors"] = []
        
        # Ensure missing_errors is a list
        if not isinstance(result["missing_errors"], list):
            logger.warning(f"missing_errors is not a list, got {type(result['missing_errors'])}")
            result["missing_errors"] = []
        
        # Convert requested errors to keys for easier lookup
        requested_keys = {}
        for error in requested_errors:
            if not isinstance(error, dict):
                logger.warning(f"Skipping non-dict error in requested_errors: {error}")
                continue
                
            error_type = error.get("type", "").upper()
            error_name = error.get("name", "")
            key = f"{error_type} - {error_name}"
            requested_keys[key] = error
        
        # Process found errors to make sure they're in the right format for regeneration
        processed_found_errors = []
        
        for error in result["found_errors"]:
            # Skip non-dict errors with warning
            if not isinstance(error, dict):
                try:
                    error_str = str(error)
                    # Try to extract error type and name from string
                    match = re.search(r'([A-Z]+)\s*-\s*([^:]+)', error_str)
                    if match:
                        error_type = match.group(1).strip()
                        error_name = match.group(2).strip()
                        processed_found_errors.append(f"{error_type} - {error_name}")
                    else:
                        logger.warning(f"Could not process non-dict error: {error_str}")
                except:
                    logger.warning(f"Could not process non-dict error: {error}")
                continue
                
            error_type = error.get("error_type", "").upper()
            error_name = error.get("error_name", "")
            
            if error_type and error_name:
                processed_found_errors.append(f"{error_type} - {error_name}")
        
        # Process missing errors to ensure they're in the right format for regeneration
        processed_missing_errors = []
        
        for error in result["missing_errors"]:
            # Skip non-dict errors with warning
            if not isinstance(error, dict):
                try:
                    error_str = str(error)
                    # Try to extract error type and name from string
                    match = re.search(r'([A-Z]+)\s*-\s*([^:]+)', error_str)
                    if match:
                        error_type = match.group(1).strip()
                        error_name = match.group(2).strip()
                        processed_missing_errors.append(f"{error_type} - {error_name}")
                    else:
                        logger.warning(f"Could not process non-dict error: {error_str}")
                except:
                    logger.warning(f"Could not process non-dict error: {error}")
                continue
                
            error_type = error.get("error_type", "").upper()
            error_name = error.get("error_name", "")
            
            if error_type and error_name:
                processed_missing_errors.append(f"{error_type} - {error_name}")
        
        # Update the result with processed data
        result["found_errors"] = processed_found_errors
        result["missing_errors"] = processed_missing_errors
        
        # Validate the "valid" field based on found vs requested errors
        result["valid"] = len(processed_missing_errors) == 0 and len(processed_found_errors) == len(requested_errors)
        
        # Store the original requested error count
        result["original_error_count"] = len(requested_errors)
        
        # Generate a feedback message
        if result["valid"]:
            result["feedback"] = f"All {len(requested_errors)} requested errors are properly implemented."
        else:
            result["feedback"] = (f"Found {len(processed_found_errors)} out of {len(requested_errors)} "
                            f"requested errors. Missing {len(processed_missing_errors)} errors.")
        
        return result
    
    