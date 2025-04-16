"""
Workflow Manager for Java Peer Review Training System.

This module provides a central manager class that integrates
all components of the workflow system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import os

from langgraph.graph import StateGraph
from state_schema import WorkflowState, ReviewAttempt

from data.json_error_repository import JsonErrorRepository
from core.code_generator import CodeGenerator
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from core.code_evaluation import CodeEvaluationAgent

from workflow.node import WorkflowNodes
from workflow.conditions import WorkflowConditions
from workflow.builder import GraphBuilder
from utils.llm_logger import LLMInteractionLogger
from utils.code_utils import generate_comparison_report

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    Manager class for the Java Code Review workflow system.
    
    This class integrates all components of the workflow system and provides
    a high-level API for interacting with the workflow.
    """
    
    def __init__(self, llm_manager):
        """
        Initialize the workflow manager with the LLM manager.
        
        Args:
            llm_manager: Manager for LLM models
        """
        self.llm_manager = llm_manager
        self.llm_logger = LLMInteractionLogger()
        
        # Initialize repositories
        self.error_repository = JsonErrorRepository()
        
        # Initialize domain objects
        self._initialize_domain_objects()
        
        # Create workflow nodes and conditions
        self.workflow_nodes = self._create_workflow_nodes()
        self.conditions = WorkflowConditions()
        
        # Build workflow graph
        self.workflow = self._build_workflow_graph()
    
    def _initialize_domain_objects(self) -> None:
        """Initialize domain objects with appropriate LLMs."""
        logger.info("Initializing domain objects for workflow")
        
        # Determine provider and check connection
        provider = self.llm_manager.provider.lower()
        connection_status = False
        
        if provider == "ollama":
            connection_status, message = self.llm_manager.check_ollama_connection()
            if not connection_status:
                logger.warning(f"Ollama connection failed: {message}")
        elif provider == "groq":
            connection_status, message = self.llm_manager.check_groq_connection()
            if not connection_status:
                logger.warning(f"Groq connection failed: {message}")
        else:
            logger.warning(f"Unknown provider: {provider}")
        
        if connection_status:
            # Initialize models for different functions
            generative_model = self._initialize_model_for_role("GENERATIVE")
            review_model = self._initialize_model_for_role("REVIEW")
            summary_model = self._initialize_model_for_role("SUMMARY")
            
            # Initialize domain objects with models
            self.code_generator = CodeGenerator(generative_model, self.llm_logger)
            self.code_evaluation = CodeEvaluationAgent(generative_model, self.llm_logger)
            self.evaluator = StudentResponseEvaluator(review_model, llm_logger=self.llm_logger)
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            # Store feedback models for generating final feedback
            self.summary_model = summary_model
            
            logger.info("Domain objects initialized with LLM models")
        else:
            # Initialize without LLMs if connection fails
            logger.warning(f"LLM connection failed. Initializing without LLMs.")
            self.code_generator = CodeGenerator(llm_logger=self.llm_logger)
            self.code_evaluation = CodeEvaluationAgent(llm_logger=self.llm_logger)
            self.evaluator = StudentResponseEvaluator(llm_logger=self.llm_logger)
            self.feedback_manager = FeedbackManager(self.evaluator)
            self.summary_model = None
    
    def _initialize_model_for_role(self, role: str):
        """
        Initialize an LLM for a specific role.
        
        Args:
            role: Role identifier (e.g., "GENERATIVE", "REVIEW")
            
        Returns:
            Initialized LLM or None if initialization fails
        """
        try:
            # Ensure GPU usage for Ollama models
            if self.llm_manager.provider.lower() == "ollama":
                # Force GPU usage
                os.environ["ENABLE_GPU"] = "true"
                self.llm_manager.force_gpu = True
                
                # Refresh GPU info
                self.llm_manager.refresh_gpu_info()
            
            # Initialize model
            return self.llm_manager.initialize_model_from_env(f"{role}_MODEL", f"{role}_TEMPERATURE")
        except Exception as e:
            logger.error(f"Error initializing {role} model: {str(e)}")
            return None
    
    def _create_workflow_nodes(self) -> WorkflowNodes:
        """
        Create workflow nodes with initialized domain objects.
        
        Returns:
            WorkflowNodes instance
        """
        logger.info("Creating workflow nodes")
        nodes = WorkflowNodes(
            self.code_generator,
            self.code_evaluation,
            self.error_repository,
            self.llm_logger
        )
        
        # Attach evaluator to nodes (needed for analyze_review_node)
        nodes.evaluator = self.evaluator
        
        return nodes
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the workflow graph using the graph builder.
        
        Returns:
            StateGraph: The constructed workflow graph
        """
        logger.info("Building workflow graph")
        builder = GraphBuilder(self.workflow_nodes)
        return builder.build_graph()
    
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """
        Get all available error categories.
        
        Returns:
            Dictionary with 'build' and 'checkstyle' categories
        """
        return self.error_repository.get_all_categories()
    
    def submit_review(self, state: WorkflowState, student_review: str) -> WorkflowState:
        """
        Submit a student review and update the state.
        
        Args:
            state: Current workflow state
            student_review: The student's review text
            
        Returns:
            Updated workflow state with analysis
        """
        logger.info(f"Submitting review for iteration {state.current_iteration}")
        
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
        updated_state = self.workflow_nodes.analyze_review_node(state)
        
        # Check if this is the last iteration or review is sufficient
        if (updated_state.current_iteration > updated_state.max_iterations or 
            updated_state.review_sufficient):
            # Generate comparison report for feedback tab
            self._generate_review_feedback(updated_state)
        
        return updated_state
    
    def _generate_review_feedback(self, state: WorkflowState) -> None:
        """
        Generate feedback for review completion.
        
        Args:
            state: Current workflow state
        """
        # Check if we have review history
        if not state.review_history:
            logger.warning("No review history found for generating feedback")
            return
            
        # Get latest review
        latest_review = state.review_history[-1]
        
        # Generate comparison report if not already generated
        if not state.comparison_report and state.code_snippet and state.code_snippet.known_problems:
            try:
                logger.info("Generating comparison report for feedback")
                state.comparison_report = generate_comparison_report(
                    state.code_snippet.known_problems,
                    latest_review.analysis
                )
            except Exception as e:
                logger.error(f"Error generating comparison report: {str(e)}")
                state.comparison_report = (
                    "# Review Feedback\n\n"
                    "There was an error generating a detailed comparison report. "
                    "Please check your review history for details."
                )
        
        # Generate review summary with LLM if available
        if not state.review_summary and self.summary_model and state.code_snippet:
            try:
                logger.info("Generating review summary with LLM")
                state.review_summary = self.feedback_manager.generate_final_feedback(
                    llm=self.summary_model,
                    include_resources=True,
                    include_visualizations=True
                )
            except Exception as e:
                logger.error(f"Error generating LLM review summary: {str(e)}")
                
                # Generate a fallback summary if LLM fails
                if latest_review and latest_review.analysis:
                    analysis = latest_review.analysis
                    identified_count = analysis.get("identified_count", 0)
                    total_problems = analysis.get("total_problems", 0)
                    identified_percentage = analysis.get("identified_percentage", 0)
                    
                    state.review_summary = (
                        f"# Review Summary\n\n"
                        f"You found {identified_count} out of {total_problems} issues "
                        f"({identified_percentage:.1f}% accuracy).\n\n"
                        f"Check the detailed analysis in the comparison report for more information."
                    )
        
        logger.info("Feedback generation complete")