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
from utils.error_tracking import enrich_error_information

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
            state.evaluation_attempts = 0
            state.evaluation_result = None
            state.code_generation_feedback = None            
            
            # Validate error categories
            if not selected_error_categories or (
                not selected_error_categories.get("build", []) and 
                not selected_error_categories.get("checkstyle", [])
            ):
                state.error = "No error categories selected. Please select at least one error category before generating code."
                return state
                               
            # Get errors from selected categories
            selected_errors, basic_problem_descriptions = self.error_repository.get_errors_for_llm(
                selected_categories=selected_error_categories,
                count=get_error_count_for_difficulty(difficulty_level),
                difficulty=difficulty_level
            )
            
            # Log detailed information about selected errors for debugging
            self._log_selected_errors(selected_errors)
            
            # Generate code with selected errors
            response = self.code_generator._generate_with_llm(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )

            # Extract both annotated and clean versions
            annotated_code, clean_code = extract_both_code_versions(response)
            
            # Enrich the error information with locations
            enhanced_errors, detailed_problems = enrich_error_information(
                    annotated_code, selected_errors
            )

            # Create code snippet object with enhanced information
            code_snippet = CodeSnippet(
                code=annotated_code,  # Store annotated version with error comments
                clean_code=clean_code,  # Store clean version without error comments
                known_problems=detailed_problems,  # Use the detailed problems
                raw_errors={
                    "build": [e for e in enhanced_errors if e["type"] == "build"],
                    "checkstyle": [e for e in enhanced_errors if e["type"] == "checkstyle"]
                },
                enhanced_errors=enhanced_errors
            )
                                    
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "review"
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
                response = self.code_generator.llm.invoke(feedback_prompt)
                metadata = {
                    "code_length": state.code_length,
                    "difficulty_level": state.difficulty_level,
                    "domain": "general",
                    "selected_errors": state.selected_error_categories
                }

                self.llm_logger.log_code_regeneration(feedback_prompt, response, metadata)
                annotated_code, clean_code = extract_both_code_versions(response)                
                
                # Get requested errors from state
                requested_errors = self._extract_requested_errors(state)
                
                # Enrich the error information 
                enhanced_errors, detailed_problems = enrich_error_information(
                    annotated_code, requested_errors
                )
                
                # Create updated code snippet
                state.code_snippet = CodeSnippet(
                    code=annotated_code,
                    clean_code=clean_code,
                    known_problems=detailed_problems,
                    raw_errors={
                        "build": [e for e in requested_errors if e.get("type") == "build"],
                        "checkstyle": [e for e in requested_errors if e.get("type") == "checkstyle"]
                    },
                    enhanced_errors=enhanced_errors
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
            
            # Evaluate the code
            evaluation_result = self.code_evaluation.evaluate_code(
                code, requested_errors
            )
            
            # Update state with evaluation results
            state.evaluation_result = evaluation_result
            state.evaluation_attempts += 1
            
            # Log evaluation results
            found_count = len(evaluation_result.get('found_errors', []))
            total_count = len(requested_errors)
            logger.info(f"Code evaluation complete: {found_count}/{total_count} errors implemented")
            
            # If evaluation passed (all errors implemented), move to review
            if evaluation_result.get("valid", False):
                state.current_step = "review"
                logger.info("All errors successfully implemented, proceeding to review")
            else:
                # Generate feedback for code regeneration
                feedback = self.code_evaluation.generate_improved_prompt(
                    code, requested_errors, evaluation_result
                )
                state.code_generation_feedback = feedback
                
                # Check if we've reached max attempts
                if state.evaluation_attempts >= state.max_evaluation_attempts:
                    # If we've reached max attempts, proceed to review anyway
                    state.current_step = "review"
                    logger.warning(f"Reached maximum evaluation attempts ({state.max_evaluation_attempts}). Proceeding to review.")
                else:
                    # Otherwise, set the step to regenerate code
                    state.current_step = "regenerate"
                    logger.info(f"Evaluation attempt {state.evaluation_attempts}: Feedback generated for regeneration")
            
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
            known_problems = state.code_snippet.known_problems
            enhanced_errors = getattr(state.code_snippet, "enhanced_errors", None)
            
            # Get the student response evaluator from the evaluator attribute
            evaluator = getattr(self, "evaluator", None)
            if not evaluator:
                state.error = "Student response evaluator not initialized"
                return state
            
            # Use the enhanced evaluation method if available
            if hasattr(evaluator, 'evaluate_review_enhanced'):
                analysis = evaluator.evaluate_review_enhanced(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    enhanced_errors=enhanced_errors
                )
            else:
                # Fall back to standard evaluation
                analysis = evaluator.evaluate_review(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    enhanced_errors=enhanced_errors
                )
            
            # Update the review with analysis
            latest_review.analysis = analysis
            
            # Check if the review is sufficient
            review_sufficient = analysis.get("review_sufficient", False)
            state.review_sufficient = review_sufficient
            
            # Generate targeted guidance if needed
            if not review_sufficient and state.current_iteration < state.max_iterations:
                targeted_guidance = self._generate_guidance(
                    evaluator=evaluator,
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    review_analysis=analysis,
                    iteration_count=state.current_iteration,
                    max_iterations=state.max_iterations,
                    enhanced_errors=enhanced_errors
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
    
    def _generate_guidance(self, evaluator, code_snippet, known_problems, student_review, 
                         review_analysis, iteration_count, max_iterations, enhanced_errors=None):
        """
        Generate targeted guidance for the student using the appropriate method.
        
        Args:
            evaluator: The student response evaluator
            code_snippet: The code being reviewed
            known_problems: List of known problems in the code
            student_review: The student's review text
            review_analysis: Analysis of the student's review
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            enhanced_errors: Optional enhanced error information
            
        Returns:
            Targeted guidance text
        """
        # Use enhanced guidance if available
        if hasattr(evaluator, 'generate_targeted_guidance_enhanced'):
            return evaluator.generate_targeted_guidance_enhanced(
                code_snippet=code_snippet,
                known_problems=known_problems,
                student_review=student_review,
                review_analysis=review_analysis,
                iteration_count=iteration_count,
                max_iterations=max_iterations,
                enhanced_errors=enhanced_errors
            )
        else:
            # Fall back to standard guidance
            return evaluator.generate_targeted_guidance(
                code_snippet=code_snippet,
                known_problems=known_problems,
                student_review=student_review,
                review_analysis=review_analysis,
                iteration_count=iteration_count,
                max_iterations=max_iterations
            )
            
    def _extract_requested_errors(self, state: WorkflowState) -> List[Dict[str, Any]]:
        """
        Extract requested errors from the state.
        
        Args:
            state: Current workflow state
            
        Returns:
            List of requested errors
        """
        requested_errors = []
        if hasattr(state.code_snippet, "raw_errors"):
            for error_type in state.code_snippet.raw_errors:
                requested_errors.extend(state.code_snippet.raw_errors[error_type])
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