"""
Workflow Builder for Java Peer Review Training System.

This module provides the GraphBuilder class for constructing
the LangGraph workflow graph with appropriate nodes and edges.
"""

import logging
from langgraph.graph import StateGraph, END

from state_schema import WorkflowState
from workflow.node import WorkflowNodes
from workflow.conditions import WorkflowConditions

# Configure logging
logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Builder for the Java Code Review workflow graph.
    
    This class is responsible for building the LangGraph graph with all necessary
    nodes and edges, including conditional edges.
    """
    
    def __init__(self, workflow_nodes: WorkflowNodes):
        """
        Initialize the graph builder with workflow nodes.
        
        Args:
            workflow_nodes: WorkflowNodes instance containing node handlers
        """
        self.workflow_nodes = workflow_nodes
        self.conditions = WorkflowConditions()
    
    def build_graph(self) -> StateGraph:
        """
        Build the complete LangGraph workflow.
        
        Returns:
            StateGraph: The constructed workflow graph
        """
        logger.info("Building Java Code Review workflow graph")
        
        # Create a new graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Add all nodes to the graph
        self._add_nodes(workflow)
        
        # Add standard edges to the graph
        self._add_standard_edges(workflow)
        
        # Add conditional edges to the graph
        self._add_conditional_edges(workflow)
        
        # Set the entry point
        workflow.set_entry_point("generate_code")
        
        logger.info("Workflow graph construction completed")
        return workflow
    
    def _add_nodes(self, workflow: StateGraph) -> None:
        """
        Add all nodes to the workflow graph.
        
        Args:
            workflow: StateGraph to add nodes to
        """
        # Define main workflow nodes
        workflow.add_node("generate_code", self.workflow_nodes.generate_code_node)
        workflow.add_node("evaluate_code", self.workflow_nodes.evaluate_code_node)
        workflow.add_node("regenerate_code", self.workflow_nodes.regenerate_code_node)
        workflow.add_node("review_code", self.workflow_nodes.review_code_node)
        workflow.add_node("analyze_review", self.workflow_nodes.analyze_review_node)
        
        logger.debug("Added all nodes to workflow graph")
    
    def _add_standard_edges(self, workflow: StateGraph) -> None:
        """
        Add standard (non-conditional) edges to the workflow graph.
        
        Args:
            workflow: StateGraph to add edges to
        """
        # Add direct edges between nodes
        workflow.add_edge("generate_code", "evaluate_code")
        workflow.add_edge("regenerate_code", "evaluate_code")
        workflow.add_edge("review_code", "analyze_review")
        workflow.add_edge("generate_summary", END)
        
        logger.debug("Added standard edges to workflow graph")
    
    def _add_conditional_edges(self, workflow: StateGraph) -> None:
        """
        Add conditional edges to the workflow graph.
        
        Args:
            workflow: StateGraph to add conditional edges to
        """
        # Add conditional edge for code evaluation
        workflow.add_conditional_edges(
            "evaluate_code",
            self.conditions.should_regenerate_or_review,
            {
                "regenerate_code": "regenerate_code",
                "review_code": "review_code"
            }
        )
        
        # Add conditional edges for review cycle
        workflow.add_conditional_edges(
            "analyze_review",
            self.conditions.should_continue_review,
            {
                "continue_review": "review_code",
                "generate_summary": "generate_summary"
            }
        )
        
        logger.debug("Added conditional edges to workflow graph")

    def export_graph_visualization(self, filename: str = "workflow_graph.png") -> str:
        """
        Export the workflow graph as a visualization image.
        
        Args:
            filename: Output filename for the graph image
            
        Returns:
            Path to the generated image file
        """
        logger.info(f"Exporting workflow graph visualization to {filename}")
        
        # Build the graph if we need a fresh instance
        workflow = self.build_graph()
        
        # Generate the graph visualization
        try:
            # Create the plot and save to file
            from langgraph.graph import plot
            graph_plot = plot(workflow)
            
            # Customize the graph for better visualization
            graph_plot.update_layout(
                title="Java Code Review Workflow",
                font=dict(size=14),
                height=800,
                width=1000,
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                margin=dict(t=40, l=20, r=20, b=20)
            )
            
            # Save the image with higher resolution
            graph_plot.write_image(filename, scale=2)
            
            logger.info(f"Successfully exported graph visualization to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting graph visualization: {str(e)}")
            
            # Try to create an alternative visualization using Mermaid
            try:
                import base64
                from IPython.display import Image
                from langchain_core.runnables.graph import MermaidDrawMethod
                
                # Generate a PNG from Mermaid
                mermaid_png = workflow.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API
                )
                
                # Save the PNG to file
                with open(filename, 'wb') as f:
                    f.write(mermaid_png)
                
                logger.info(f"Successfully exported Mermaid visualization to {filename}")
                return filename
            except Exception as mermaid_error:
                logger.error(f"Error creating Mermaid visualization: {str(mermaid_error)}")
                return None

    