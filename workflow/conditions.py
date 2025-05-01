"""
Workflow Conditions for Java Peer Review Training System.

This module contains the conditional logic for determining
which paths to take in the LangGraph workflow.
"""

import logging
from typing import Dict, Any, List, Optional
from state_schema import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowConditions:
    """
    Conditional logic for the Java Code Review workflow.
    
    This class contains all the conditional functions used to determine
    the next step in the workflow based on the current state.
    """
    
    @staticmethod
    def should_regenerate_or_review(state: WorkflowState) -> str:
        """
        Determine if we should regenerate code or move to review.
        Enhanced to handle Groq-specific issues.
        
        Args:
            state: Current workflow state
            
        Returns:
            "regenerate_code" if we need to regenerate code based on evaluation feedback
            "review_code" if the code is valid or we've reached max attempts
        """
        logger.debug(f"Deciding workflow path with state: step={state.current_step}, "
                f"valid={state.evaluation_result and state.evaluation_result.get('valid', False)}, "
                f"attempts={getattr(state, 'evaluation_attempts', 0)}/{getattr(state, 'max_evaluation_attempts', 3)}")
        
        # Check if current step is explicitly set to regenerate
        if state.current_step == "regenerate":
            logger.info("Path decision: regenerate_code (explicit current_step)")
            return "regenerate_code"
        
        # Check if LLM provider is Groq - may need special handling
        is_groq_provider = False
        if hasattr(state, 'provider') and state.provider == "groq":
            is_groq_provider = True
            logger.info("Using Groq provider, applying special handling")
        
        # Check if evaluation passed - all errors implemented correctly
        if state.evaluation_result and state.evaluation_result.get("valid", False):
            logger.info("Path decision: review_code (evaluation passed)")
            return "review_code"
        
        # Check for EXTRACTION_FAILED special marker in missing errors (Groq fallback)
        if state.evaluation_result and "missing_errors" in state.evaluation_result:
            missing_errors = state.evaluation_result["missing_errors"]
            if "EXTRACTION_FAILED" in missing_errors:
                # Force regeneration for extraction failures with Groq
                if hasattr(state, 'evaluation_attempts') and state.evaluation_attempts < getattr(state, 'max_evaluation_attempts', 3):
                    logger.info("Path decision: regenerate_code (extraction failed, forcing regeneration)")
                    return "regenerate_code"
        
        # IMPORTANT: Check for missing errors that need fixing
        if state.evaluation_result and state.evaluation_result.get("missing_errors", []):
            # Only regenerate if we haven't reached max attempts
            if hasattr(state, 'evaluation_attempts') and state.evaluation_attempts < getattr(state, 'max_evaluation_attempts', 3):
                logger.info(f"Path decision: regenerate_code (evaluation found missing errors)")
                return "regenerate_code"
        
        # Special case for Groq: Be more aggressive with regeneration for Groq provider
        if is_groq_provider and hasattr(state, 'evaluation_attempts') and state.evaluation_attempts < getattr(state, 'max_evaluation_attempts', 3):
            # If we're using Groq and haven't reached max attempts, check if the evaluation might have failed
            if not state.evaluation_result or not state.evaluation_result.get("found_errors"):
                logger.info("Path decision: regenerate_code (Groq provider with potential evaluation issues)")
                return "regenerate_code"
        
        # If no evaluation result yet, check if we can do evaluation 
        if state.code_snippet:
            logger.info("Path decision: review_code (code exists but no evaluation result)")
            return "review_code"
        
        # Default case - need to generate code first
        logger.info("Path decision: review_code (default case)")
        return "review_code"
    
    @staticmethod
    def should_continue_review(state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        
        This is used for the conditional edge from the analyze_review node.
        
        Args:
            state: Current workflow state
            
        Returns:
            "continue_review" if more review iterations are needed
            "generate_summary" if the review is sufficient or max iterations reached
        """
        logger.debug(f"Deciding review path with state: iteration={state.current_iteration}/{state.max_iterations}, "
                   f"sufficient={state.review_sufficient}")
        
        # Check if we've reached max iterations
        if state.current_iteration > state.max_iterations:
            logger.info(f"Review path decision: generate_summary (max iterations reached: {state.current_iteration})")
            return "generate_summary"
        
        # Check if the review is sufficient
        if state.review_sufficient:
            logger.info("Review path decision: generate_summary (review sufficient)")
            return "generate_summary"
        
        # Otherwise, continue reviewing
        logger.info(f"Review path decision: continue_review (iteration {state.current_iteration}, not sufficient)")
        return "continue_review"