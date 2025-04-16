"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph
with improved organization and maintainability.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
import os
import random
import re
from langgraph.graph import StateGraph, END
from state_schema import WorkflowState, CodeSnippet, ReviewAttempt

# Import domain-specific components
from core.code_generator import CodeGenerator
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from core.code_evaluation import CodeEvaluationAgent

from data.json_error_repository import JsonErrorRepository
from llm_manager import LLMManager

from typing import Dict, Any, Optional, List, Union, Tuple

from utils.code_utils import extract_both_code_versions, get_error_count_for_difficulty
from utils.error_tracking import enrich_error_information
from utils.llm_logger import LLMInteractionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class WorkflowNodes:
    """
    Handles node operations in the Java Code Review workflow.
    
    This class contains all node implementations for the workflow,
    extracted for better organization and maintainability.
    """
    
    def __init__(self, code_generator, code_evaluation, error_repository, evaluator, llm_logger):
        """Initialize the workflow nodes handler."""
        self.code_generator = code_generator
        self.code_evaluation = code_evaluation
        self.error_repository = error_repository
        self.evaluator = evaluator
        self.llm_logger = llm_logger
    
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Generate Java code with errors node with enhanced debugging for all modes."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            state.evaluation_attempts = 0
            state.evaluation_result = None
            state.code_generation_feedback = None            
            
            # Check if we have valid selected categories
            if not selected_error_categories or (
                not selected_error_categories.get("build", []) and 
                not selected_error_categories.get("checkstyle", [])
            ):
                # Instead of using defaults, require explicit selection
                state.error = "No error categories selected. Please select at least one error category before generating code."
                return state
                               
            # Get errors from selected categories
            selected_errors, basic_problem_descriptions = self.error_repository.get_errors_for_llm(
                selected_categories=selected_error_categories,
                count=get_error_count_for_difficulty(difficulty_level),
                difficulty=difficulty_level
            )
            
            # Enhanced debugging: Print detailed information about selected errors            
            if selected_errors:
                self._log_selected_errors(selected_errors)
            
            response = self.code_generator._generate_with_llm(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )

            # Now returns both annotated and clean versions
            annotated_code, clean_code = extract_both_code_versions(response)
            
            # Enrich the error information 
            enhanced_errors, detailed_problems = enrich_error_information(
                    annotated_code, selected_errors
            )

            # Create code snippet object with enhanced information
            code_snippet = CodeSnippet(
                code=annotated_code,  # Store annotated version with error comments
                clean_code=clean_code,  # Store clean version without error comments
                known_problems=detailed_problems,  # Use the detailed problems instead
                raw_errors={
                    "build": [e for e in enhanced_errors if e["type"] == "build"],
                    "checkstyle": [e for e in enhanced_errors if e["type"] == "checkstyle"]
                },
                # Add the enhanced error information
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
        Uses the feedback from code evaluation to improve code generation.
        Handles both string and AIMessage responses.
        """
        try:
            logger.info(f"Starting enhanced code regeneration (Attempt {state.evaluation_attempts})")
            
            # Use the code generation feedback to generate improved code
            feedback_prompt = state.code_generation_feedback
            
            # Log the prompt for debugging
            logger.info("Using feedback prompt for regeneration")
            
            # Generate code with feedback prompt
            if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
                # Log provider information for debugging
                if hasattr(self.code_generator.llm, 'provider'):
                    logger.info(f"Regenerating code with provider: {self.code_generator.llm.provider}")
                
                # Generate response using LLM
                response = self.code_generator.llm.invoke(feedback_prompt)
                
                # Log response type for debugging
                logger.info(f"Regeneration response type: {type(response).__name__}")
                
                # Log the regeneration
                metadata = {
                    "code_length": "text",
                    "difficulty_level": "text",
                    "domain": "text",
                    "selected_errors": "text",
                    "attempt": state.evaluation_attempts
                }
                self.llm_logger.log_code_regeneration(feedback_prompt, response, metadata)
                
                # Extract code from the response, handling both string and AIMessage responses
                annotated_code, clean_code = extract_both_code_versions(response)
                
                # If we couldn't extract clean code, use the strip function
                if not clean_code and annotated_code:
                    clean_code = strip_error_annotations(annotated_code)
                    
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
        Acts as a dedicated node in the LangGraph workflow.
        Uses LLM for more accurate evaluation when available.
        """
        try:
            logger.info("Starting code evaluation node")
            
            # Get code snippet from state
            if not state.code_snippet:
                state.error = "No code snippet available for evaluation"
                return state
                
            # Get the clean code (without annotations)
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
        """Review code node - this is a placeholder since user input happens in the UI."""
        # This node is primarily a placeholder since the actual review is submitted via the UI
        state.current_step = "review"
        return state
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze student review node."""
        try:
            # Get the latest review
            if not state.review_history:
                state.error = "No review submitted to analyze"
                return state
                    
            latest_review = state.review_history[-1]
            student_review = latest_review.student_review
            
            # Get code snippet
            if not state.code_snippet:
                state.error = "No code snippet available"
                return state
                    
            code_snippet = state.code_snippet.code
            known_problems = state.code_snippet.known_problems
            enhanced_errors = getattr(state.code_snippet, "enhanced_errors", None)
            
            # Use the enhanced evaluation method if available
            if hasattr(self.evaluator, 'evaluate_review_enhanced'):
                analysis = self.evaluator.evaluate_review_enhanced(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    enhanced_errors=enhanced_errors
                )
            else:
                # Fall back to standard evaluation
                analysis = self.evaluator.evaluate_review(
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
                # Use enhanced guidance if available
                targeted_guidance = self._generate_guidance(
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
            
    def _extract_requested_errors(self, state: WorkflowState) -> List[Dict[str, Any]]:
        """Extract requested errors from the state."""
        requested_errors = []
        if hasattr(state.code_snippet, "raw_errors"):
            for error_type in state.code_snippet.raw_errors:
                requested_errors.extend(state.code_snippet.raw_errors[error_type])
        return requested_errors
        
    def _generate_guidance(self, code_snippet, known_problems, student_review, 
                           review_analysis, iteration_count, max_iterations, enhanced_errors=None):
        """Generate targeted guidance using the appropriate method."""
        # Use enhanced guidance if available
        if hasattr(self.evaluator, 'generate_targeted_guidance_enhanced'):
            return self.evaluator.generate_targeted_guidance_enhanced(
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
            return self.evaluator.generate_targeted_guidance(
                code_snippet=code_snippet,
                known_problems=known_problems,
                student_review=student_review,
                review_analysis=review_analysis,
                iteration_count=iteration_count,
                max_iterations=max_iterations
            )
            
    def _log_selected_errors(self, selected_errors):
        """Log detailed information about selected errors for debugging."""
        print("\n--- DETAILED ERROR LISTING ---")
        for i, error in enumerate(selected_errors, 1):
            print(f"  {i}. Type: {error.get('type', 'Unknown')}")
            print(f"     Name: {error.get('name', 'Unknown')}")
            print(f"     Category: {error.get('category', 'Unknown')}")
            print(f"     Description: {error.get('description', 'Unknown')}")
            if 'implementation_guide' in error:
                print(f"     Implementation Guide: {error.get('implementation_guide', '')[:100]}..." 
                      if len(error.get('implementation_guide', '')) > 100 
                      else error.get('implementation_guide', ''))
            print()

class WorkflowConditions:
    """
    Handles conditional logic for the Java Code Review workflow.
    
    This class contains conditional functions for determining the next
    step in the workflow based on the current state.
    """
    
    @staticmethod
    def should_regenerate_or_review(state: WorkflowState) -> str:
        """
        Determine if we should regenerate code or move to review.
        This is used for the conditional edge from the evaluate_code node.
        
        Returns:
            "regenerate_code" if we need to regenerate code based on evaluation feedback
            "review_code" if the code is valid or we've reached max attempts
        """
        # Check if current step is explicitly set to regenerate
        if state.current_step == "regenerate":
            return "regenerate_code"
        
        # Check if evaluation passed
        if state.evaluation_result and state.evaluation_result.get("valid", False):
            return "review_code"
        
        # Check if we've reached max attempts
        if hasattr(state, 'evaluation_attempts') and state.evaluation_attempts >= state.max_evaluation_attempts:
            return "review_code"
        
        # Default to regenerate if we have an evaluation result but it's not valid
        if state.evaluation_result:
            return "regenerate_code"
        
        # If no evaluation result yet, move to review
        return "review_code"
    
    @staticmethod
    def should_continue_review(state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        This is used for the conditional edge from the analyze_review node.
        
        Returns:
            "continue_review" if more review iterations are needed
            "generate_summary" if the review is sufficient or max iterations reached
        """
        # Check if we've reached max iterations
        if state.current_iteration > state.max_iterations:
            return "generate_summary"
        
        # Check if the review is sufficient
        if state.review_sufficient:
            return "generate_summary"
        
        # Otherwise, continue reviewing
        return "continue_review"

class JavaCodeReviewGraph:
    """
    LangGraph implementation of the Java Code Review workflow.
    
    This class organizes the code review workflow as a LangGraph graph
    with improved internal structure and error handling.
    """    
    def __init__(self, llm_manager: LLMManager = None):
        """Initialize the graph with domain components."""
        # Initialize LLM Manager if not provided
        self.llm_manager = llm_manager or LLMManager()
        
        # Ensure provider is correctly set from environment variable if it exists
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and env_provider.lower() != self.llm_manager.provider:
            logger.warning(f"Provider mismatch: Environment variable LLM_PROVIDER={env_provider}, but llm_manager.provider={self.llm_manager.provider}")
            logger.info(f"Updating provider to match environment: {env_provider}")
            self.llm_manager.provider = env_provider.lower()
        
        # Log which provider is currently selected
        logger.info(f"Initializing JavaCodeReviewGraph with provider: {self.llm_manager.provider}")
        
        # Initialize repositories
        self.error_repository = JsonErrorRepository()
        
        # Initialize domain objects
        self._initialize_domain_objects()
        
        # Create the graph
        self.workflow = self._build_graph()
    
    def _initialize_domain_objects(self):
        """Initialize domain objects with LLMs if available based on selected provider."""
        # Initialize LLM logger
        self.llm_logger = LLMInteractionLogger()
        logger.info("Initialized LLM Interaction Logger")
        
        # Check which provider is currently selected
        current_provider = self.llm_manager.provider.lower()
        logger.info(f"Current LLM provider: {current_provider}")
        
        # Variables to track connection status
        connection_status = False
        connection_message = ""
        
        # Check connection based on provider
        if current_provider == "ollama":
            connection_status, connection_message = self.llm_manager.check_ollama_connection()
            if not connection_status:
                logger.warning(f"Connection to Ollama failed: {connection_message}")
        elif current_provider == "groq":
            connection_status, connection_message = self.llm_manager.check_groq_connection()
            if not connection_status:
                logger.warning(f"Connection to Groq API failed: {connection_message}")
        else:
            logger.error(f"Unsupported provider: {current_provider}")
        
        if connection_status:
            # Use environment variables to create models with correct provider
            logger.info(f"Initializing models with provider: {current_provider}")
            
            # Initialize models using environment variables
            generative_model = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
            review_model = self.llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
            
            # Log model information
            if generative_model:
                logger.info(f"Generative model initialized: {type(generative_model).__name__}")
            if review_model:
                logger.info(f"Review model initialized: {type(review_model).__name__}")
            
            # Initialize domain objects with models and logger
            self.code_generator = CodeGenerator(generative_model, self.llm_logger) if generative_model else CodeGenerator(llm_logger=self.llm_logger)
            self.code_evaluation = CodeEvaluationAgent(generative_model, self.llm_logger) if generative_model else CodeEvaluationAgent(llm_logger=self.llm_logger)
            self.evaluator = StudentResponseEvaluator(review_model, llm_logger=self.llm_logger) if review_model else StudentResponseEvaluator(llm_logger=self.llm_logger)
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            # Initialize workflow nodes
            self.workflow_nodes = WorkflowNodes(
                self.code_generator,
                self.code_evaluation,
                self.error_repository,
                self.evaluator,
                self.llm_logger
            )
            
            # Initialize conditions
            self.conditions = WorkflowConditions()
        else:
            # Connection failed - provide provider-specific error message
            if current_provider == "ollama":
                logger.error(f"Connection to Ollama failed: {connection_message}. Please check if Ollama is running.")
            elif current_provider == "groq":
                logger.error(f"Connection to Groq API failed: {connection_message}. Please check your API key and connection.")
            else:
                logger.error(f"Unsupported provider: {current_provider}")
            
            # Initialize with empty components anyway to avoid errors
            self.code_generator = CodeGenerator(llm_logger=self.llm_logger)
            self.code_evaluation = CodeEvaluationAgent(llm_logger=self.llm_logger)
            self.evaluator = StudentResponseEvaluator(llm_logger=self.llm_logger)
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            # Initialize workflow nodes with empty components
            self.workflow_nodes = WorkflowNodes(
                self.code_generator,
                self.code_evaluation,
                self.error_repository,
                self.evaluator,
                self.llm_logger
            )
            
            # Initialize conditions
            self.conditions = WorkflowConditions()
    
    def verify_provider_status(self):
        """
        Verify the current provider status and log detailed information.
        Useful for debugging provider-related issues.
        
        Returns:
            bool: True if provider is properly configured, False otherwise
        """
        current_provider = self.llm_manager.provider.lower()
        logger.info(f"Verifying LLM provider status: {current_provider}")
        
        # Check if provider is supported
        if current_provider not in ["ollama", "groq"]:
            logger.error(f"Unsupported provider: {current_provider}")
            return False
        
        # Verify provider-specific configuration
        if current_provider == "ollama":
            connection_status, message = self.llm_manager.check_ollama_connection()
            if connection_status:
                # Get available models
                try:
                    models = self.llm_manager.get_available_models()
                    model_names = [model.get("id") for model in models if model.get("pulled", False)]
                    logger.info(f"Available Ollama models: {', '.join(model_names) if model_names else 'None'}")
                    return True
                except Exception as e:
                    logger.error(f"Error listing Ollama models: {str(e)}")
                    return False
            else:
                logger.error(f"Ollama connection failed: {message}")
                return False
        
        elif current_provider == "groq":
            connection_status, message = self.llm_manager.check_groq_connection()
            if connection_status:
                # Check API key and available models
                try:
                    groq_api_key = self.llm_manager.groq_api_key
                    api_key_mask = f"{groq_api_key[:4]}...{groq_api_key[-4:]}" if len(groq_api_key) > 8 else "Not set properly"
                    logger.info(f"Groq API key: {api_key_mask}")
                    
                    available_models = self.llm_manager.groq_available_models
                    logger.info(f"Available Groq models: {', '.join(available_models)}")
                    return True
                except Exception as e:
                    logger.error(f"Error verifying Groq configuration: {str(e)}")
                    return False
            else:
                logger.error(f"Groq connection failed: {message}")
                return False
        
        return False
    
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Delegate to workflow nodes."""
        return self.workflow_nodes.generate_code_node(state)
    
    def regenerate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Delegate to workflow nodes."""
        return self.workflow_nodes.regenerate_code_node(state)
    
    def evaluate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Delegate to workflow nodes."""
        return self.workflow_nodes.evaluate_code_node(state)
    
    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """Delegate to workflow nodes."""
        return self.workflow_nodes.review_code_node(state)
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """Delegate to workflow nodes."""
        return self.workflow_nodes.analyze_review_node(state)
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with code evaluation and regeneration."""
        # Create a new graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Define nodes
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("evaluate_code", self.evaluate_code_node) 
        workflow.add_node("regenerate_code", self.regenerate_code_node)
        workflow.add_node("review_code", self.review_code_node)
        workflow.add_node("analyze_review", self.analyze_review_node)      
        
        # Define edges
        workflow.add_edge("generate_code", "evaluate_code")  # First go to evaluation
        workflow.add_edge("regenerate_code", "evaluate_code")  # Regeneration also goes to evaluation
        
        # Conditional edge based on evaluation result
        workflow.add_conditional_edges(
            "evaluate_code",
            self.should_regenerate_or_review,
            {
                "regenerate_code": "regenerate_code",
                "review_code": "review_code"
            }
        )
        
        workflow.add_edge("review_code", "analyze_review")
        
        # Conditional edges from analyze_review
        workflow.add_conditional_edges(
            "analyze_review",
            self.should_continue_review,
            {
                "continue_review": "review_code",
                "generate_summary": "generate_summary"
            }
        )
        
        workflow.add_edge("generate_summary", END)
        
        # Set the entry point
        workflow.set_entry_point("generate_code")
        
        return workflow
    
    def should_regenerate_or_review(self, state: WorkflowState) -> str:
        """Delegate to conditions."""
        return self.conditions.should_regenerate_or_review(state)
    
    def should_continue_review(self, state: WorkflowState) -> str:
        """Delegate to conditions."""
        return self.conditions.should_continue_review(state)
        
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """Get all available error categories."""
        return self.error_repository.get_all_categories()
    
    def submit_review(self, state: WorkflowState, student_review: str) -> WorkflowState:
        """
        Submit a student review and update the state.
        This method is called from the UI when a student submits a review.
        """
        # Create a new review attempt
        review_attempt = ReviewAttempt(
            student_review=student_review,
            iteration_number=state.current_iteration,
            analysis={},
            targeted_guidance=None
        )
        
        # Add to review history
        state.review_history.append(review_attempt)
        
        # Run the state through the analyze_review node
        updated_state = self.analyze_review_node(state)
        
        return updated_state