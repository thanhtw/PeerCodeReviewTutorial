






"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph,
using a modular architecture for better maintainability.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
from typing import Dict, Any, Optional, List

from state_schema import WorkflowState
from workflow import WorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class JavaCodeReviewGraph:
    """
    LangGraph implementation of the Java Code Review workflow.
    
    This class serves as a facade to the modular workflow system,
    maintaining backward compatibility with the existing API.
    """
    
    def __init__(self, llm_manager=None):
        """
        Initialize the graph with domain components.
        
        Args:
            llm_manager: Optional LLMManager for managing language models
        """
        # Initialize the workflow manager
        self.llm_manager = llm_manager
        self.workflow_manager = WorkflowManager(llm_manager)
        
        # Set up workflow and error repository references for compatibility
        self.workflow = self.workflow_manager.workflow
        self.error_repository = self.workflow_manager.error_repository
    
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate Java code with errors node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with generated code
        """
        return self.workflow_manager.workflow_nodes.generate_code_node(state)
    
    def regenerate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Regenerate code based on evaluation feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with regenerated code
        """
        return self.workflow_manager.workflow_nodes.regenerate_code_node(state)
    
    def evaluate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate generated code to ensure it contains the requested errors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with evaluation results
        """
        return self.workflow_manager.workflow_nodes.evaluate_code_node(state)
    
    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Review code node - placeholder since user input happens in the UI.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        return self.workflow_manager.workflow_nodes.review_code_node(state)
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze student review node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with review analysis
        """
        return self.workflow_manager.workflow_nodes.analyze_review_node(state)
    
    def should_regenerate_or_review(self, state: WorkflowState) -> str:
        """
        Determine if we should regenerate code or move to review.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step name
        """
        return self.workflow_manager.workflow_nodes.conditions.should_regenerate_or_review(state)
    
    def should_continue_review(self, state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step name
        """
        return self.workflow_manager.workflow_nodes.conditions.should_continue_review(state)
    
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """
        Get all available error categories.
        
        Returns:
            Dictionary with 'build' and 'checkstyle' categories
        """
        return self.workflow_manager.get_all_error_categories()
    
    def submit_review(self, state: WorkflowState, student_review: str) -> WorkflowState:
        """
        Submit a student review and update the state.
        
        Args:
            state: Current workflow state
            student_review: The student's review text
            
        Returns:
            Updated workflow state with analysis
        """
        return self.workflow_manager.submit_review(state, student_review)