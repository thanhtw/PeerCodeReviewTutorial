"""
Workflow Nodes for Java Peer Review Training System.

This module contains the node implementations for the LangGraph workflow,
separating node logic from graph construction for better maintainability.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional

from state_schema import WorkflowState, CodeSnippet
from utils.code_utils import extract_both_code_versions, get_error_count_for_difficulty

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowNodes:
    """
    Node implementations for the Java Code Review workflow.
    
    This class contains all node handlers that process state transitions
    in the LangGraph workflow, extracted for better separation of concerns.
    """
    
    def __init__(self, code_generator, code_evaluation, error_repository, llm_logger):
        """
        Initialize workflow nodes with required components.
        
        Args:
            code_generator: Component for generating Java code with errors
            code_evaluation: Component for evaluating generated code quality
            error_repository: Repository for accessing Java error data
            llm_logger: Logger for tracking LLM interactions
        """
        self.code_generator = code_generator
        self.code_evaluation = code_evaluation
        self.error_repository = error_repository
        self.llm_logger = llm_logger
    
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate Java code with errors based on selected parameters.
        Ensures exact match between selected and generated errors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with generated code
        """
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            selected_specific_errors = state.selected_specific_errors
            
            # Reset state for a fresh generation
            state.evaluation_attempts = 0
            state.evaluation_result = None
            state.code_generation_feedback = None            
            
            # Determine whether we're using specific errors or categories
            using_specific_errors = len(selected_specific_errors) > 0
            
            # Get appropriate errors based on selection mode
            if using_specific_errors:
                # Using specific errors mode - IMPORTANT: Use the exact selected errors without modification
                if not selected_specific_errors:
                    state.error = "No specific errors selected. Please select at least one error before generating code."
                    return state
                    
                logger.info(f"Using specific errors mode with {len(selected_specific_errors)} errors")
                # Use the selected errors directly without applying count filtering
                selected_errors = selected_specific_errors
                # Store the original requested error count
                original_error_count = len(selected_errors)
            else:
                # Using category-based selection mode
                if not selected_error_categories or (
                    not selected_error_categories.get("build", []) and 
                    not selected_error_categories.get("checkstyle", [])
                ):
                    state.error = "No error categories selected. Please select at least one error category before generating code."
                    return state
                            
                logger.info(f"Using category-based mode with categories: {selected_error_categories}")
                
                # Get exact number based on difficulty
                required_error_count = get_error_count_for_difficulty(difficulty_level)
                selected_errors, _ = self.error_repository.get_errors_for_llm(
                    selected_categories=selected_error_categories,
                    count=required_error_count,
                    difficulty=difficulty_level
                )
                
                # Make sure we have the right number of errors
                if len(selected_errors) < required_error_count:
                    logger.warning(f"Got fewer errors ({len(selected_errors)}) than requested ({required_error_count})")
                    # Don't modify the count in this case - use what we have
                    original_error_count = len(selected_errors)
                elif len(selected_errors) > required_error_count:
                    logger.warning(f"Got more errors ({len(selected_errors)}) than requested ({required_error_count})")
                    # Trim to exactly the required count
                    selected_errors = selected_errors[:required_error_count]
                    original_error_count = required_error_count
                else:
                    original_error_count = required_error_count
            
            # Log detailed information about selected errors for debugging
            self._log_selected_errors(selected_errors)
            logger.info(f"Final error count for generation: {len(selected_errors)}")
            
            # Generate code with selected errors - ensure clear expectations for the LLM
            # Explicitly include the count in the prompt to emphasize the requirement
            response = self.code_generator._generate_with_llm(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )

            # Extract both annotated and clean versions
            annotated_code, clean_code = extract_both_code_versions(response)

            # Create code snippet object
            code_snippet = CodeSnippet(
                code=annotated_code,  # Store annotated version with error comments
                clean_code=clean_code,  # Store clean version without error comments
                raw_errors={
                    "build": [e for e in selected_errors if e["type"].lower() == "build"],
                    "checkstyle": [e for e in selected_errors if e["type"].lower() == "checkstyle"]
                },
                expected_error_count=original_error_count  # Store the original error count in the code snippet
            )
                                    
            # Update state with the original error count for consistency
            state.original_error_count = original_error_count
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "evaluate"  # Set to evaluate instead of review to ensure proper workflow
            return state
                    
        except Exception as e:           
            logger.error(f"Error generating code: {str(e)}", exc_info=True)
            state.error = f"Error generating code: {str(e)}"
            return state

    def regenerate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Regenerate code based on evaluation feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with regenerated code
        """
        try:
            logger.info(f"Starting enhanced code regeneration (Attempt {state.evaluation_attempts})")
            
            # Use the code generation feedback to generate improved code
            feedback_prompt = state.code_generation_feedback
            
            # Generate code with feedback prompt
            if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
                # Log the regeneration prompt before sending to LLM
                metadata = {
                    "code_length": state.code_length,
                    "difficulty_level": state.difficulty_level,
                    "domain": "general",
                    "selected_errors": state.selected_error_categories,
                    "attempt": state.evaluation_attempts,
                    "max_attempts": state.max_evaluation_attempts
                }
                
                # Log the prompt before it's sent to the LLM
                self.llm_logger.log_regeneration_prompt(feedback_prompt, metadata)
                
                # Generate the code
                response = self.code_generator.llm.invoke(feedback_prompt)
                
                # Log the full regeneration with response
                self.llm_logger.log_code_regeneration(feedback_prompt, response, metadata)
                
                # Process the response
                annotated_code, clean_code = extract_both_code_versions(response)                
                
                # Get requested errors from state
                requested_errors = self._extract_requested_errors(state)
                
                # Create updated code snippet
                state.code_snippet = CodeSnippet(
                    code=annotated_code,
                    clean_code=clean_code,
                    raw_errors={
                        "build": [e for e in requested_errors if e.get("type") == "build"],
                        "checkstyle": [e for e in requested_errors if e.get("type") == "checkstyle"]
                    }
                )
                
                # Move to evaluation step again
                state.current_step = "evaluate"
                logger.info(f"Code regenerated successfully on attempt {state.evaluation_attempts}")
                
                return state
            else:
                # If no LLM available, fall back to standard generation
                logger.warning("No LLM available for regeneration. Falling back to standard generation.")
                return self.generate_code_node(state)
            
        except Exception as e:                 
            logger.error(f"Error regenerating code: {str(e)}", exc_info=True)
            state.error = f"Error regenerating code: {str(e)}"
            return state
        
    def evaluate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate generated code to ensure it contains the requested errors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with evaluation results
        """
        try:
            logger.info("Starting code evaluation node")
            
            # Validate code snippet
            if not state.code_snippet:
                state.error = "No code snippet available for evaluation"
                return state
                    
            # Get the code with annotations
            code = state.code_snippet.code
            
            # Get requested errors from state
            requested_errors = self._extract_requested_errors(state)
            requested_count = len(requested_errors)
            
            # Ensure we're using the original error count for consistency
            original_error_count = state.original_error_count
            if original_error_count == 0 and hasattr(state.code_snippet, 'expected_error_count'):
                # If not set in state, try to get it from code snippet
                original_error_count = state.code_snippet.expected_error_count
                # Update state with this count
                state.original_error_count = original_error_count
                
            # If we still don't have it, use the requested count
            if original_error_count == 0:
                original_error_count = requested_count
                state.original_error_count = original_error_count
                
            logger.info(f"Evaluating code for {original_error_count} expected errors")
            
            # Evaluate the code
            raw_evaluation_result = self.code_evaluation.evaluate_code(
                code, requested_errors
            )
            
            # IMPORTANT: Ensure evaluation_result is a dictionary
            if not isinstance(raw_evaluation_result, dict):
                logger.error(f"Expected dict for evaluation_result, got {type(raw_evaluation_result)}")
                # Create a default dictionary with the necessary structure
                evaluation_result = {
                    "found_errors": [],
                    "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                    for error in requested_errors],
                    "valid": False,
                    "feedback": f"Error in evaluation. Please ensure the code contains all {original_error_count} requested errors.",
                    "original_error_count": original_error_count  # Add original count for consistency
                }
            else:
                evaluation_result = raw_evaluation_result
                # Add the original error count to the evaluation result
                evaluation_result["original_error_count"] = original_error_count
            
            # Update state with evaluation results
            state.evaluation_result = evaluation_result
            state.evaluation_attempts += 1
            
            # Log evaluation results
            found_count = len(evaluation_result.get('found_errors', []))
            missing_count = len(evaluation_result.get('missing_errors', []))
            logger.info(f"Code evaluation complete: {found_count}/{original_error_count} errors implemented, {missing_count} missing")
            
            # Check if there are extra errors beyond what was requested
            extra_errors = evaluation_result.get('extra_errors', [])
            has_extra_errors = len(extra_errors) > 0
            
            # Generate feedback for code regeneration regardless of valid status
            # This ensures we always have feedback ready if needed
            feedback = None
            
            # If we have missing errors or extra errors, we need to regenerate the code
            needs_regeneration = missing_count > 0 or has_extra_errors
            
            # If we have extra errors, use the updated regeneration function that handles extras
            if has_extra_errors:
                logger.warning(f"Found {len(extra_errors)} extra errors beyond the {original_error_count} requested")
                
                # Use the version of regeneration prompt that handles extra errors
                if hasattr(self.code_evaluation, 'generate_improved_prompt_with_extras'):
                    feedback = self.code_evaluation.generate_improved_prompt_with_extras(
                        code, requested_errors, evaluation_result
                    )
                else:
                    # Import the updated function if not available as a method
                    from utils.code_utils import create_regeneration_prompt
                    
                    # Prepare data for regeneration with emphasis on having exactly the right errors
                    feedback = create_regeneration_prompt(
                        code=code,
                        domain=self._infer_domain_from_code(code),
                        missing_errors=evaluation_result.get('missing_errors', []),
                        found_errors=evaluation_result.get('found_errors', []),
                        requested_errors=requested_errors,
                        extra_errors=extra_errors
                    )
            # Missing errors but no extra errors
            elif missing_count > 0:
                logger.warning(f"Missing {missing_count} out of {original_error_count} requested errors")
                
                # Use standard regeneration prompt but enhance it for clarity
                if hasattr(self.code_evaluation, 'generate_improved_prompt'):
                    feedback = self.code_evaluation.generate_improved_prompt(
                        code, requested_errors, evaluation_result
                    )
                else:
                    # Import the updated function if not available as a method
                    from utils.code_utils import create_regeneration_prompt
                    
                    # Use the regeneration prompt with emphasis on adding missing errors
                    feedback = create_regeneration_prompt(
                        code=code,
                        domain=self._infer_domain_from_code(code),
                        missing_errors=evaluation_result.get('missing_errors', []),
                        found_errors=evaluation_result.get('found_errors', []),
                        requested_errors=requested_errors,
                        extra_errors=[]
                    )
            else:
                # No missing or extra errors - we're good!
                logger.info(f"All {original_error_count} requested errors implemented correctly")
                
                # Still create a feedback prompt in case we need it later
                from utils.code_utils import create_regeneration_prompt
                feedback = create_regeneration_prompt(
                    code=code,
                    domain=self._infer_domain_from_code(code),
                    missing_errors=[],
                    found_errors=evaluation_result.get('found_errors', []),
                    requested_errors=requested_errors,
                    extra_errors=[]
                )
                    
            state.code_generation_feedback = feedback
          
            # IMPROVED DECISION LOGIC: Prioritize fixing missing errors over max attempts
            # If evaluation passed (all errors implemented with exact count)
            if evaluation_result.get("valid", False):
                state.current_step = "review"
                logger.info("All errors successfully implemented, proceeding to review")
            elif needs_regeneration and state.evaluation_attempts < state.max_evaluation_attempts:
                # If we have missing errors or extra errors and haven't reached max attempts, regenerate
                state.current_step = "regenerate"
                if missing_count > 0:
                    logger.info(f"Found {missing_count} missing errors, proceeding to regeneration")
                if has_extra_errors:
                    logger.info(f"Found {len(extra_errors)} extra errors, proceeding to regeneration")
            else:
                # Otherwise, we've either reached max attempts or have no more missing errors
                state.current_step = "review"
                if state.evaluation_attempts >= state.max_evaluation_attempts:
                    logger.warning(f"Reached maximum evaluation attempts ({state.max_evaluation_attempts}). Proceeding to review.")
                else:
                    logger.info("No missing errors to fix, proceeding to review")
            
            return state
            
        except Exception as e:
            logger.error(f"Error evaluating code: {str(e)}", exc_info=True)
            state.error = f"Error evaluating code: {str(e)}"
            return state

    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Review code node - this is a placeholder since user input happens in the UI.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        # This node is primarily a placeholder since the actual review is submitted via the UI
        state.current_step = "review"
        return state
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze student review and provide feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with review analysis
        """
        try:
            # Validate review history
            if not state.review_history:
                state.error = "No review submitted to analyze"
                return state
                    
            latest_review = state.review_history[-1]
            student_review = latest_review.student_review
            
            # Validate code snippet
            if not state.code_snippet:
                state.error = "No code snippet available"
                return state
                    
            code_snippet = state.code_snippet.code
            
            # Use evaluation result to extract problem information
            # But modify to use the original error count for consistent metrics
            known_problems = []
            original_error_count = state.original_error_count
            
            if state.evaluation_result and 'found_errors' in state.evaluation_result:
                known_problems = state.evaluation_result.get('found_errors', [])
            
            # Get the student response evaluator from the evaluator attribute
            evaluator = getattr(self, "evaluator", None)
            if not evaluator:
                state.error = "Student response evaluator not initialized"
                return state
            
            # Use the standard evaluation method
            analysis = evaluator.evaluate_review(
                code_snippet=code_snippet,
                known_problems=known_problems,
                student_review=student_review
            )
            
            # IMPORTANT: Update the analysis with the original error count
            # This ensures consistent metrics in the UI
            if original_error_count > 0:
                # Store the found problem count and original count
                found_problems_count = len(known_problems)
                identified_count = analysis.get("identified_count", 0)
                
                # IMPORTANT FIX: Override total_problems to use original_error_count
                analysis["total_problems"] = original_error_count
                analysis["original_error_count"] = original_error_count
                
                # Recalculate percentages based on original count
                analysis["identified_percentage"] = (identified_count / original_error_count) * 100
                analysis["accuracy_percentage"] = (identified_count / original_error_count) * 100
                
                logger.info(f"Updated review analysis: {identified_count}/{original_error_count} " +
                        f"({analysis['identified_percentage']:.1f}%) [Found problems: {found_problems_count}]")
            
            # Update the review with analysis
            latest_review.analysis = analysis
            
            # Check if the review is sufficient
            review_sufficient = analysis.get("review_sufficient", False)
            state.review_sufficient = review_sufficient
            
            # Generate targeted guidance if needed
            if not review_sufficient and state.current_iteration < state.max_iterations:
                targeted_guidance = evaluator.generate_targeted_guidance(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    review_analysis=analysis,
                    iteration_count=state.current_iteration,
                    max_iterations=state.max_iterations
                )
                latest_review.targeted_guidance = targeted_guidance
            
            # Increment iteration count
            state.current_iteration += 1
            
            # Update state
            state.current_step = "analyze"
            
            return state
        
        except Exception as e:
            logger.error(f"Error analyzing review: {str(e)}", exc_info=True)
            state.error = f"Error analyzing review: {str(e)}"
            return state
    
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
            
    def _extract_requested_errors(self, state: WorkflowState) -> List[Dict[str, Any]]:
        """
        Extract requested errors from the state with improved error handling and type safety.
        
        Args:
            state: Current workflow state
            
        Returns:
            List of requested errors
        """
        requested_errors = []
        
        # First check if code_snippet exists
        if not hasattr(state, 'code_snippet') or state.code_snippet is None:
            logger.warning("No code snippet in state for extracting requested errors")
            return requested_errors
        
        # Check if raw_errors exists and is a dictionary
        if hasattr(state.code_snippet, "raw_errors"):
            raw_errors = state.code_snippet.raw_errors
            
            # Type check for raw_errors
            if not isinstance(raw_errors, dict):
                logger.warning(f"Expected dict for raw_errors, got {type(raw_errors)}")
                return requested_errors
            
            # Extract errors from each type
            for error_type, errors in raw_errors.items():
                # Type check for errors list
                if not isinstance(errors, list):
                    logger.warning(f"Expected list for errors of type {error_type}, got {type(errors)}")
                    continue
                    
                # Type check each error and add to requested_errors
                for error in errors:
                    if not isinstance(error, dict):
                        logger.warning(f"Expected dict for error, got {type(error)}")
                        continue
                    
                    # Make sure the error has required fields
                    if "type" not in error:
                        error["type"] = error_type  # Use the key as type if not specified
                    
                    if "name" not in error and "error_name" in error:
                        error["name"] = error["error_name"]  # Use error_name as name if available
                    
                    if "name" not in error and "check_name" in error:
                        error["name"] = error["check_name"]  # Use check_name as name if available
                    
                    # Only add the error if it has a name
                    if "name" in error:
                        requested_errors.append(error)
        
        # Alternative method: Check selected_specific_errors
        elif hasattr(state, 'selected_specific_errors') and state.selected_specific_errors:
            # Check if it's a list
            if isinstance(state.selected_specific_errors, list):
                # Filter out non-dict entries
                for error in state.selected_specific_errors:
                    if isinstance(error, dict) and "name" in error and "type" in error:
                        requested_errors.append(error)
        
        # If we still don't have any errors, check selected_error_categories
        if not requested_errors and hasattr(state, 'selected_error_categories'):
            # Check if it's a dict
            if isinstance(state.selected_error_categories, dict):
                # This doesn't give us specific errors, but we can log that we found categories
                logger.info("Found selected_error_categories but no specific errors")
        
        logger.info(f"Extracted {len(requested_errors)} requested errors")
        return requested_errors
        
    def _log_selected_errors(self, selected_errors: List[Dict[str, Any]]) -> None:
        """
        Log detailed information about selected errors for debugging.
        
        Args:
            selected_errors: List of selected errors
        """
        if selected_errors:
            logger.debug("\n--- DETAILED ERROR LISTING ---")
            for i, error in enumerate(selected_errors, 1):
                logger.debug(f"  {i}. Type: {error.get('type', 'Unknown')}")
                logger.debug(f"     Name: {error.get('name', 'Unknown')}")
                logger.debug(f"     Category: {error.get('category', 'Unknown')}")
                logger.debug(f"     Description: {error.get('description', 'Unknown')}")
                if 'implementation_guide' in error:
                    guide = error.get('implementation_guide', '')
                    logger.debug(f"     Implementation Guide: {guide[:100]}..." 
                        if len(guide) > 100 else guide)