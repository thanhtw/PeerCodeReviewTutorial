"""
Feedback Tab UI module for Java Peer Review Training System.

This module provides the functions for rendering the feedback and analysis tab.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def render_feedback_tab(workflow, feedback_display_ui):
    """Render the feedback and analysis tab with enhanced visuals."""
    state = st.session_state.workflow_state
    
    if not state.comparison_report and not state.review_summary:
        st.info("Please submit your review in the 'Submit Review' tab first.")
        return
    
    # Reset callback with confirmation
    def handle_reset():
        # Create a confirmation dialog
        if st.session_state.get("confirm_reset", False) or st.button("Confirm Reset", key="confirm_reset_btn"):
            # Create a new workflow state
            st.session_state.workflow_state = workflow.__class__()
            
            # Reset active tab
            st.session_state.active_tab = 0
            
            # Reset confirmation flag
            if "confirm_reset" in st.session_state:
                del st.session_state.confirm_reset
            
            # Rerun the app
            st.rerun()
    
    # Get the latest review analysis
    latest_review = state.review_history[-1] if state.review_history else None
    latest_analysis = latest_review.analysis if latest_review else None
    
    # Convert review history to the format expected by FeedbackDisplayUI
    review_history = []
    for review in state.review_history:
        review_history.append({
            "iteration_number": review.iteration_number,
            "student_review": review.student_review,
            "review_analysis": review.analysis
        })
    
    # Display feedback results
    feedback_display_ui.render_results(
        comparison_report=state.comparison_report,
        review_summary=state.review_summary,
        review_analysis=latest_analysis,
        review_history=review_history,
        on_reset_callback=handle_reset
    )