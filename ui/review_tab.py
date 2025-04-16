"""
Review Tab UI module for Java Peer Review Training System.

This module provides the functions for rendering the review submission tab
and handling student reviews.
"""

import streamlit as st
import logging
import time
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_student_review(workflow, student_review: str):
    """
    Process a student review with progress indicator and improved error handling.
    
    Args:
        workflow: The JavaCodeReviewGraph workflow instance
        student_review: The student's review text
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Show progress during analysis
    with st.status("Processing your review...", expanded=True) as status:
        try:
            # Get current state
            if not hasattr(st.session_state, 'workflow_state'):
                status.update(label="Error: Workflow state not initialized", state="error")
                st.session_state.error = "Please generate a code problem first"
                return False
                
            state = st.session_state.workflow_state
            
            # Check if code snippet exists
            if not state.code_snippet:
                status.update(label="Error: No code snippet available", state="error")
                st.session_state.error = "Please generate a code problem first"
                return False
            
            # Check if student review is empty
            if not student_review.strip():
                status.update(label="Error: Review cannot be empty", state="error")
                st.session_state.error = "Please enter your review before submitting"
                return False
            
            # Store the current review in session state for display consistency
            current_iteration = state.current_iteration
            st.session_state[f"submitted_review_{current_iteration}"] = student_review
            
            # Update status
            status.update(label="Analyzing your review...", state="running")
            
            # Log submission attempt
            logger.info(f"Submitting review (iteration {current_iteration}): {student_review[:100]}...")
            
            # Submit the review and update the state
            updated_state = workflow.submit_review(state, student_review)
            
            # Check for errors
            if updated_state.error:
                status.update(label=f"Error: {updated_state.error}", state="error")
                st.session_state.error = updated_state.error
                logger.error(f"Error during review analysis: {updated_state.error}")
                return False
            
            # Update session state
            st.session_state.workflow_state = updated_state
            
            # Log successful analysis
            logger.info(f"Review analysis complete for iteration {current_iteration}")
            
            # Check if we should generate summary
            if workflow.should_continue_review(updated_state) == "generate_summary":
                status.update(label="Generating final feedback...", state="running")
                
                try:                 
                    # Move to the analysis tab
                    st.session_state.active_tab = 2                    
                    status.update(label="Analysis complete! Moving to Feedback tab...", state="complete")
                except Exception as e:
                    error_msg = f"Error generating final feedback: {str(e)}"
                    logger.error(error_msg)
                    status.update(label=error_msg, state="error")
                    st.session_state.error = error_msg
                    return False
            else:
                status.update(label="Analysis complete!", state="complete")
            
            # Force a rerun to update the UI
            time.sleep(0.5)  # Short delay to ensure the status message is visible
            st.rerun()
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing student review: {str(e)}"
            logger.error(error_msg)
            status.update(label=error_msg, state="error")
            st.session_state.error = error_msg
            return False

def render_review_tab(workflow, code_display_ui):
    """
    Render the review tab UI with automatic tab switching on completion.
    
    Args:
        workflow: JavaCodeReviewGraph workflow
        code_display_ui: CodeDisplayUI instance for displaying code
    """
    st.subheader("Review Java Code")
    
    # Check if we have a code snippet to review
    if not hasattr(st.session_state, 'code_snippet') or not st.session_state.code_snippet:
        st.info("No code has been generated yet. Please go to the 'Generate Problem' tab first.")
        return
    
    # Display the code
    code_display_ui.render_code_display(st.session_state.code_snippet)
    
    # Get current review state
    current_iteration = getattr(st.session_state, 'current_iteration', 1)
    max_iterations = getattr(st.session_state, 'max_iterations', 3)
    
    # Get the latest review if available
    latest_review = None
    targeted_guidance = None
    review_analysis = None
    
    if hasattr(st.session_state, 'review_history') and st.session_state.review_history:
        if len(st.session_state.review_history) > 0:
            latest_review = st.session_state.review_history[-1]
            targeted_guidance = getattr(latest_review, 'targeted_guidance', None)
            review_analysis = getattr(latest_review, 'analysis', {})
    
    # Only allow submission if we're under the max iterations
    if current_iteration <= max_iterations:
        # Get the current student review (empty for first iteration)
        student_review = ""
        if latest_review is not None:
            student_review = latest_review.student_review
        
        # Define submission callback
        def on_submit_review(review_text):
            logger.info(f"Submitting review (iteration {current_iteration})")
            
            # Update session state with reviewed state
            state = workflow.submit_review(st.session_state, review_text)
            
            # Update session state with the new state
            for key, value in state.__dict__.items():
                setattr(st.session_state, key, value)
            
            # Check if this was the last iteration or review is sufficient
            if current_iteration >= max_iterations or state.review_sufficient:
                logger.info("Review process complete, switching to feedback tab")
                # Switch to feedback tab (index 2)
                st.session_state.active_tab = 2
                # Force rerun to update UI
                st.rerun()
        
        # Render review input with current state
        code_display_ui.render_review_input(
            student_review=student_review,
            on_submit_callback=on_submit_review,
            iteration_count=current_iteration,
            max_iterations=max_iterations,
            targeted_guidance=targeted_guidance,
            review_analysis=review_analysis
        )
    else:
        # If we've reached max iterations, display a message and auto-switch to feedback tab
        st.warning(f"You have completed all {max_iterations} review iterations. View feedback in the next tab.")
        
        # Automatically switch to feedback tab if not already there
        if st.session_state.active_tab != 2:  # 2 is the index of the feedback tab
            st.session_state.active_tab = 2
            st.rerun()