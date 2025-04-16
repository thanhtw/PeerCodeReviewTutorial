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

from utils.error_validation import validate_code_errors
from utils.llm_logger import LLMInteractionLogger
from utils.code_utils import create_evaluation_prompt, create_regeneration_prompt

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
        Evaluate Java code to check for requested errors with enhanced response processing.
        
        Args:
            code: The Java code to evaluate
            requested_errors: List of errors that should be included in the code
            
        Returns:
            Evaluation results with found and missing errors
        """
        # Default result if no evaluation can be performed
        default_result = {
            "found_errors": [],
            "missing_errors": requested_errors,
            "valid": False,
            "feedback": f"Could not evaluate code. Please ensure the code contains all {len(requested_errors)} requested errors."
        }
        
        # Check if LLM is available for evaluation
        if not self.llm:
            logger.warning("No LLM available for code evaluation")
            return default_result
        
        # Convert requested errors to a format easier to track
        requested_error_keys = []
        for error in requested_errors:
            error_type = error.get("type", "").upper()
            error_name = error.get("name", "")
            requested_error_keys.append(f"{error_type} - {error_name}")
        
        # Create evaluation prompt
        prompt = create_evaluation_prompt(code, requested_errors)
        
        # Log the preparation information
        logger.info(f"Evaluating code for {len(requested_errors)} requested errors")
        
        try:
            # Generate the evaluation using the LLM
            logger.info("Sending code to LLM for evaluation")
            response = self.llm.invoke(prompt)
            
            # Process the response to ensure it's a string
            processed_response = process_llm_response(response)
            
            # Log the evaluation
            if self.llm_logger:
                metadata = {
                    "code_length": len(code.splitlines()),
                    "requested_errors_count": len(requested_errors),
                    "requested_errors": requested_error_keys
                }
                self.llm_logger.log_code_evaluation(prompt, processed_response, metadata)
            
            # Extract JSON from the response
            evaluation_result = self._extract_json_from_response(processed_response)
            
            # If extraction failed, return default result
            if not evaluation_result:
                logger.warning("Failed to extract JSON from evaluation response")
                return default_result
            
            # Post-process the evaluation result
            return self._process_evaluation_result(evaluation_result, requested_errors)
            
        except Exception as e:
            logger.error(f"Error evaluating code: {str(e)}")
            return default_result
    
    def _evaluate_with_llm(self, code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate generated code using an LLM to identify if requested errors are present.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.llm:
            logger.warning("No LLM provided for evaluation, falling back to regex-based validation")
            return validate_code_errors(code, requested_errors)
        
        # Use the optimized prompt creation function
        prompt = create_evaluation_prompt(code, requested_errors)
        
        
        try:
            # Get response from LLM
            response = self.llm.invoke(prompt)
            # Metadata for logging
            metadata = {
                "requested_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" for error in requested_errors],
                "code_length": len(code.splitlines()),
            }           
            # Log the interaction
            self.llm_logger.log_code_evaluation(prompt, response, metadata)
            
            # Extract JSON from response
            import re
            import json
            
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from LLM response")
                    return validate_code_errors(code, requested_errors)
            else:
                # Try to find any JSON object in the response
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        analysis = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from LLM response")
                        return validate_code_errors(code, requested_errors)
                else:
                    logger.error("No JSON found in LLM response")
                    return validate_code_errors(code, requested_errors)
            
            # Process the analysis results
            found_errors = []
            missing_errors = []
            error_locations = {}
            
            # Extract found errors
            for error in analysis.get("found_errors", []):
                error_type = error.get("error_type", "")
                error_name = error.get("error_name", "")
                error_key = f"{error_type} - {error_name}"
                
                found_errors.append(error_key)
                error_locations[error_key] = error.get("line_number", 0)
            
            # Extract missing errors
            for error in analysis.get("missing_errors", []):
                error_type = error.get("error_type", "")
                error_name = error.get("error_name", "")
                error_key = f"{error_type} - {error_name}"
                
                missing_errors.append(error_key)
            
            # Check if we're missing any errors that weren't explicitly mentioned
            all_requested_keys = [f"{error.get('type', '').upper()} - {error.get('name', '')}" for error in requested_errors]
            for key in all_requested_keys:
                if key not in found_errors and key not in missing_errors:
                    missing_errors.append(key)
            
            # Create the validation result
            validation_results = {
                "valid": analysis.get("valid", False),
                "found_errors": found_errors,
                "missing_errors": missing_errors,
                "error_locations": error_locations,
                "llm_feedback": analysis.get("feedback", ""),
                "detailed_analysis": analysis  # Keep the full LLM analysis for detailed feedback
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error evaluating code with LLM: {str(e)}")
            
            # Log the error
            error_metadata = {**metadata, "error": str(e)}
            self.llm_logger.log_code_evaluation(prompt, f"ERROR: {str(e)}", error_metadata)
            
            # Fall back to regex-based validation
            return validate_code_errors(code, requested_errors)
    
    def _generate_feedback(self, code: str, requested_errors: List[Dict[str, Any]], 
                         validation_results: Dict[str, Any]) -> str:
        """
        Generate detailed feedback on the implementation of errors.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            validation_results: Results from validation
            
        Returns:
            Detailed feedback string
        """
        # If LLM feedback is available, use it
        if "llm_feedback" in validation_results and validation_results["llm_feedback"]:
            return validation_results["llm_feedback"]
        
        # Otherwise, use the original feedback generation logic
        lines = code.splitlines()
        feedback = []
        
        # Provide feedback on correctly implemented errors
        if validation_results["found_errors"]:
            feedback.append("Successfully implemented errors:")
            for error_key in validation_results["found_errors"]:
                line_num = validation_results["error_locations"].get(error_key, 0)
                line_content = lines[line_num-1] if 0 < line_num <= len(lines) else "Unknown"
                feedback.append(f"- {error_key} (Line {line_num}: '{line_content.strip()}')")
        
        # Provide feedback on missing errors
        if validation_results["missing_errors"]:
            feedback.append("\nErrors that need implementation:")
            for error_key in validation_results["missing_errors"]:
                # Find the corresponding error details
                error_details = None
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        error_details = error
                        break
                
                if error_details:
                    implementation_guide = error_details.get("implementation_guide", "No implementation guide available")
                    feedback.append(f"- {error_key}")
                    feedback.append(f"  Implementation guide: {implementation_guide}")
                else:
                    feedback.append(f"- {error_key} (Details not available)")
        
        # Overall assessment
        if validation_results["valid"]:
            feedback.append("\nAll requested errors have been successfully implemented in the code.")
        else:
            found_count = len(validation_results["found_errors"])
            total_count = len(requested_errors)
            feedback.append(f"\nImplemented {found_count} out of {total_count} requested errors "
                          f"({found_count/total_count*100:.1f}%).")
            
        return "\n".join(feedback)
    
    def _generate_suggestions(self, code: str, requested_errors: List[Dict[str, Any]], 
                            validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate specific suggestions for implementing missing errors.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            validation_results: Results from validation
            
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # If we have detailed analysis from LLM, use it to generate better suggestions
        if "detailed_analysis" in validation_results:
            detailed_analysis = validation_results["detailed_analysis"]
            missing_errors_analysis = detailed_analysis.get("missing_errors", [])
            
            for error_analysis in missing_errors_analysis:
                error_type = error_analysis.get("error_type", "")
                error_name = error_analysis.get("error_name", "")
                explanation = error_analysis.get("explanation", "")
                
                error_key = f"{error_type} - {error_name}"
                
                suggestion = {
                    "error_key": error_key,
                    "suggestions": [explanation]
                }
                
                # Try to find the corresponding error details for implementation guide
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        implementation_guide = error.get("implementation_guide", "")
                        if implementation_guide:
                            suggestion["suggestions"].append(f"Implementation guide: {implementation_guide}")
                        break
                
                suggestions.append(suggestion)
        else:
            # Fall back to original suggestion generation
            for error_key in validation_results["missing_errors"]:
                # Find the corresponding error details
                error_details = None
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        error_details = error
                        break
                
                if not error_details:
                    continue
                    
                error_type = error_details.get("type", "").lower()
                error_name = error_details.get("name", "")
                
                suggestion = {
                    "error_key": error_key,
                    "suggestions": []
                }
                
                # Get implementation guide if available
                implementation_guide = error_details.get("implementation_guide", "")
                if implementation_guide:
                    suggestion["suggestions"].append(f"Follow implementation guide: {implementation_guide}")
                
                # Add generic suggestion based on error name
                suggestion["suggestions"].append(
                    f"Look for ways to introduce a {error_name} error in the code"
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
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
        
        # Get missing and found errors
        missing_errors = evaluation.get("missing_errors", [])
        found_errors = evaluation.get("found_errors", [])
        
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
        Extract JSON data from LLM response with improved regex handling.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted JSON data or None if extraction fails
        """
        # Check if response is None or empty
        if not response:
            return None
        
        # Try to find JSON block with various patterns
        patterns = [
            r'```json\s*([\s\S]*?)```',  # JSON in code block
            r'```\s*({[\s\S]*?})\s*```',  # Any JSON in code block
            r'({[\s\S]*?"found_errors"[\s\S]*?})',  # JSON with found_errors field
            r'({[\s\S]*?"valid"[\s\S]*?})',  # JSON with valid field
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
        
        # If all extraction methods fail, return None
        return None
    
    def _process_evaluation_result(self, result: Dict[str, Any], 
                                requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and enhance the evaluation result.
        
        Args:
            result: Raw evaluation result from LLM
            requested_errors: List of requested errors
            
        Returns:
            Processed evaluation result
        """
        # Ensure all expected fields exist
        if "found_errors" not in result:
            result["found_errors"] = []
        if "missing_errors" not in result:
            result["missing_errors"] = []
        
        # Convert requested errors to keys for easier lookup
        requested_keys = {}
        for error in requested_errors:
            error_type = error.get("type", "").upper()
            error_name = error.get("name", "")
            key = f"{error_type} - {error_name}"
            requested_keys[key] = error
        
        # Validate found errors against requested errors
        validated_found = []
        for error in result["found_errors"]:
            error_type = error.get("error_type", "").upper()
            error_name = error.get("error_name", "")
            key = f"{error_type} - {error_name}"
            
            if key in requested_keys:
                # Add original error data for reference
                error["original_data"] = requested_keys[key]
                validated_found.append(error)
        
        # Find missing errors by comparing found errors with requested errors
        found_keys = set()
        for error in validated_found:
            error_type = error.get("error_type", "").upper()
            error_name = error.get("error_name", "")
            found_keys.add(f"{error_type} - {error_name}")
        
        missing_errors = []
        for key, error in requested_keys.items():
            if key not in found_keys:
                missing_errors.append({
                    "error_type": error.get("type", "").upper(),
                    "error_name": error.get("name", ""),
                    "explanation": f"No implementation of this error found in the code"
                })
        
        # Update the result with validated data
        result["found_errors"] = validated_found
        result["missing_errors"] = missing_errors
        
        # Validate the "valid" field based on found vs requested errors
        result["valid"] = len(missing_errors) == 0
        
        # Generate a feedback message
        if result["valid"]:
            result["feedback"] = f"All {len(requested_errors)} requested errors are properly implemented."
        else:
            result["feedback"] = (f"Found {len(validated_found)} out of {len(requested_errors)} "
                               f"requested errors. Missing {len(missing_errors)} errors.")
        
        return result
    
    def generate_improved_prompt(self, code: str, requested_errors: List[Dict[str, Any]], 
                              evaluation_result: Dict[str, Any]) -> str:
        """
        Generate an improved prompt for code regeneration.
        
        Args:
            code: The original Java code
            requested_errors: List of requested errors
            evaluation_result: Evaluation result with found and missing errors
            
        Returns:
            Improved prompt for regeneration
        """
        # Extract found errors
        found_errors = []
        for error in evaluation_result.get("found_errors", []):
            error_type = error.get("error_type", "").upper()
            error_name = error.get("error_name", "")
            found_errors.append(f"{error_type} - {error_name}")
        
        # Extract missing errors
        missing_errors = []
        for error in evaluation_result.get("missing_errors", []):
            error_type = error.get("error_type", "").upper()
            error_name = error.get("error_name", "")
            missing_errors.append(f"{error_type} - {error_name}")
        
        # Get a domain estimate from the code
        domain = self._estimate_domain(code)
        
        # Create a regeneration prompt
        return create_regeneration_prompt(
            code=code,
            domain=domain,
            missing_errors=missing_errors,
            found_errors=found_errors,
            requested_errors=requested_errors
        )
    
    def _estimate_domain(self, code: str) -> str:
        """
        Estimate the domain of the code based on class and variable names.
        
        Args:
            code: The Java code
            
        Returns:
            Estimated domain name
        """
        # Common domains to check for
        domains = {
            "user_management": ["User", "Account", "Profile", "Authentication"],
            "file_processing": ["File", "Stream", "Reader", "Writer"],
            "data_validation": ["Validator", "Validation", "Verify", "Sanitize"],
            "calculation": ["Calculator", "Math", "Compute", "Calculate"],
            "inventory_system": ["Inventory", "Product", "Stock", "Item"],
            "notification_service": ["Notification", "Alert", "Message", "Email"],
            "logging": ["Logger", "Log", "Audit", "Monitor"],
            "banking": ["Bank", "Account", "Transaction", "Payment"],
            "e_commerce": ["Cart", "Order", "Product", "Payment"],
            "student_management": ["Student", "Course", "Grade", "Enrollment"]
        }
        
        # Count occurrences of domain keywords
        domain_counts = {domain: 0 for domain in domains}
        
        for domain, keywords in domains.items():
            for keyword in keywords:
                domain_counts[domain] += len(re.findall(r'\b' + keyword + r'\b', code))
        
        # Find the domain with the highest count
        max_count = 0
        best_domain = "general"
        
        for domain, count in domain_counts.items():
            if count > max_count:
                max_count = count
                best_domain = domain
        
        # If no significant match, return "general"
        if max_count < 2:
            return "general"
        
        return best_domain
    