"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph
by leveraging the modular components from the workflow package.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
from typing import Dict, List, Any, Optional

from state_schema import WorkflowState

# Import workflow components
from workflow.manager import WorkflowManager
from workflow.conditions import WorkflowConditions

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
    maintaining backward compatibility with the existing API while
    delegating actual implementation to the workflow package components.
    """
    
    def __init__(self, llm_manager=None):
        """
        Initialize the graph with domain components.
        
        Args:
            llm_manager: Optional LLMManager for managing language models
        """
        # Initialize the workflow manager with LLM manager
        self.llm_manager = llm_manager
        self.workflow_manager = WorkflowManager(llm_manager)
        
        # Set up references to workflow components for backward compatibility
        self.workflow = self.workflow_manager.workflow
        self.error_repository = self.workflow_manager.error_repository
        
        # Get references to workflow nodes and conditions
        self.workflow_nodes = self.workflow_manager.workflow_nodes
        self.conditions = WorkflowConditions()
    
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate Java code with errors node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with generated code
        """
        # Delegate to workflow nodes implementation
        return self.workflow_nodes.generate_code_node(state)
    
    def regenerate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Regenerate code based on evaluation feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with regenerated code
        """
        # Delegate to workflow nodes implementation
        return self.workflow_nodes.regenerate_code_node(state)
    
    def evaluate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate generated code to ensure it contains the requested errors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with evaluation results
        """
        # Delegate to workflow nodes implementation
        return self.workflow_nodes.evaluate_code_node(state)
    
    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Review code node - placeholder since user input happens in the UI.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        # Delegate to workflow nodes implementation
        return self.workflow_nodes.review_code_node(state)
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze student review node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with review analysis
        """
        # Delegate to workflow nodes implementation
        return self.workflow_nodes.analyze_review_node(state)
    
    def should_regenerate_or_review(self, state: WorkflowState) -> str:
        """
        Determine if we should regenerate code or move to review.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step name
        """
        # Delegate to workflow conditions implementation
        return self.conditions.should_regenerate_or_review(state)
    
    def should_continue_review(self, state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step name
        """
        # Delegate to workflow conditions implementation
        return self.conditions.should_continue_review(state)
    
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """
        Get all available error categories.
        
        Returns:
            Dictionary with 'build' and 'checkstyle' categories
        """
        # Delegate to error repository - use the correct method name
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
        # Delegate to workflow manager implementation
        return self.workflow_manager.submit_review(state, student_review)