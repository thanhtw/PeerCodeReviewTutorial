"""
Workflow Manager for Java Peer Review Training System.

This module provides a central manager class that integrates
all components of the workflow system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from langgraph.graph import StateGraph
from state_schema import WorkflowState, ReviewAttempt

from data.json_error_repository import JsonErrorRepository
from core.code_generator import CodeGenerator
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from core.code_evaluation import CodeEvaluationAgent

from workflow.nodes import WorkflowNodes
from workflow.builder import GraphBuilder
from utils.llm_logger import LLMInteractionLogger

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
        
        # Create workflow nodes and build graph
        self.workflow_nodes = self._create_workflow_nodes()
        self.workflow = self._build_workflow_graph()
    
    def _initialize_domain_objects(self) -> None:
        """Initialize domain objects with appropriate LLMs."""
        logger.info("Initializing domain objects for workflow")
        
        # Check Ollama connection
        connection_status, message = self.llm_manager.check_ollama_connection()
        
        if connection_status:
            # Initialize models for different functions
            generative_model = self._initialize_model_for_role("GENERATIVE")
            review_model = self._initialize_model_for_role("REVIEW")
            
            # Initialize domain objects with models
            self.code_generator = CodeGenerator(generative_model, self.llm_logger)
            self.code_evaluation = CodeEvaluationAgent(generative_model, self.llm_logger)
            self.evaluator = StudentResponseEvaluator(review_model, llm_logger=self.llm_logger)
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            logger.info("Domain objects initialized with LLM models")
        else:
            # Initialize without LLMs if connection fails
            logger.warning(f"Ollama connection failed: {message}. Initializing without LLMs.")
            self.code_generator = CodeGenerator(llm_logger=self.llm_logger)
            self.code_evaluation = CodeEvaluationAgent(llm_logger=self.llm_logger)
            self.evaluator = StudentResponseEvaluator(llm_logger=self.llm_logger)
            self.feedback_manager = FeedbackManager(self.evaluator)
    
    def _initialize_model_for_role(self, role: str):
        """
        Initialize an LLM for a specific role.
        
        Args:
            role: Role identifier (e.g., "GENERATIVE", "REVIEW")
            
        Returns:
            Initialized LLM or None if initialization fails
        """
        try:
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
        
        return updated_state