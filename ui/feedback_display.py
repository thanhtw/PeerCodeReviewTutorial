"""
Feedback Display UI module for Java Peer Review Training System.

This module provides the FeedbackDisplayUI class for displaying feedback on student reviews.
"""

import streamlit as st
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedbackDisplayUI:
    """
    UI Component for displaying feedback on student reviews.
    
    This class handles displaying analysis results, review history,
    and feedback on student reviews.
    """
    
    def render_results(self, 
                      comparison_report: str = None,
                      review_summary: str = None,
                      review_analysis: Dict[str, Any] = None,
                      review_history: List[Dict[str, Any]] = None,
                      on_reset_callback: Callable[[], None] = None) -> None:
        """
        Render the analysis results and feedback with improved review visibility.
        
        Args:
            comparison_report: Comparison report text
            review_summary: Review summary text
            review_analysis: Analysis of student review
            review_history: History of review iterations
            on_reset_callback: Callback function when reset button is clicked
        """
        if not comparison_report and not review_summary and not review_analysis:
            st.info("No analysis results available. Please submit your review in the 'Submit Review' tab first.")
            return
        
        # First show performance summary metrics at the top
        if review_history and len(review_history) > 0 and review_analysis:
            self._render_performance_summary(review_analysis, review_history)
        
        # Display the comparison report
        if comparison_report:
            st.subheader("Educational Feedback:")
            st.markdown(
                f'<div class="comparison-report">{comparison_report}</div>',
                unsafe_allow_html=True
            )
        
        # Always show review history for better visibility
        if review_history and len(review_history) > 0:
            st.subheader("Your Review:")
            
            # First show the most recent review prominently
            if review_history:
                latest_review = review_history[-1]
                review_analysis = latest_review.get("review_analysis", {})
                iteration = latest_review.get("iteration_number", 0)
                
                st.markdown(f"#### Your Final Review (Attempt {iteration})")
                
                # Format the review text with syntax highlighting
                st.markdown("```text\n" + latest_review.get("student_review", "") + "\n```")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Issues Found", 
                        f"{review_analysis.get('identified_count', 0)} of {review_analysis.get('total_problems', 0)}",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Accuracy", 
                        f"{review_analysis.get('accuracy_percentage', 0):.1f}%",
                        delta=None
                    )
                with col3:
                    false_positives = len(review_analysis.get('false_positives', []))
                    st.metric(
                        "False Positives", 
                        false_positives,
                        delta=None
                    )
            
            # Show earlier reviews in an expander if there are multiple
            if len(review_history) > 1:
                with st.expander("Review History", expanded=False):
                    tabs = st.tabs([f"Attempt {rev.get('iteration_number', i+1)}" for i, rev in enumerate(review_history)])
                    
                    for i, (tab, review) in enumerate(zip(tabs, review_history)):
                        with tab:
                            review_analysis = review.get("review_analysis", {})
                            st.markdown("```text\n" + review.get("student_review", "") + "\n```")
                            
                            st.write(f"**Found:** {review_analysis.get('identified_count', 0)} of "
                                    f"{review_analysis.get('total_problems', 0)} issues "
                                    f"({review_analysis.get('accuracy_percentage', 0):.1f}% accuracy)")
        
        # Display analysis details in an expander
        if review_summary or review_analysis:
            with st.expander("Detailed Analysis", expanded=True):
                tabs = st.tabs(["Identified Issues", "Missed Issues"])
                
                with tabs[0]:  # Identified Issues
                    self._render_identified_issues(review_analysis)
                
                with tabs[1]:  # Missed Issues
                    self._render_missed_issues(review_analysis)

        # Start over button
        st.markdown("---")            
            
    
    def _render_performance_summary(self, review_analysis: Dict[str, Any], review_history: List[Dict[str, Any]]):
        """Render performance summary metrics and charts using the consistent original error count"""
        st.subheader("Review Performance Summary")
        
        # Create performance metrics using the original error count if available
        col1, col2, col3 = st.columns(3)
        
        # Get the correct total_problems count from original_error_count if available
        original_error_count = review_analysis.get("original_error_count", 0)
        if original_error_count <= 0:
            # Fallback to total_problems if original_error_count is not available
            original_error_count = review_analysis.get("total_problems", 0)
        
        # If still zero, make a final check with the found and missed counts
        if original_error_count <= 0:
            identified_count = review_analysis.get("identified_count", 0)
            missed_count = len(review_analysis.get("missed_problems", []))
            original_error_count = identified_count + missed_count
        
        # Now calculate the accuracy using the original count for consistency
        identified_count = review_analysis.get("identified_count", 0)
        accuracy = (identified_count / original_error_count * 100) if original_error_count > 0 else 0
        
        with col1:
            st.metric(
                "Overall Accuracy", 
                f"{accuracy:.1f}%",
                delta=None
            )
            
        with col2:
            st.metric(
                "Issues Identified", 
                f"{identified_count}/{original_error_count}",
                delta=None
            )
            
        with col3:
            false_positives = len(review_analysis.get("false_positives", []))
            st.metric(
                "False Positives", 
                f"{false_positives}",
                delta=None
            )
            
        # Create a progress chart if multiple iterations
        if len(review_history) > 1:
            # Extract data for chart
            iterations = []
            identified_counts = []
            accuracy_percentages = []
            
            for review in review_history:
                analysis = review.get("review_analysis", {})
                iterations.append(review.get("iteration_number", 0))
                
                # Use consistent error count for all iterations
                review_identified = analysis.get("identified_count", 0)
                identified_counts.append(review_identified)
                
                # Calculate accuracy consistently
                review_accuracy = (review_identified / original_error_count * 100) if original_error_count > 0 else 0
                accuracy_percentages.append(review_accuracy)
                    
            # Create a DataFrame for the chart
            chart_data = pd.DataFrame({
                "Iteration": iterations,
                "Issues Found": identified_counts,
                "Accuracy (%)": accuracy_percentages
            })
            
            # Display the chart with two y-axes
            st.subheader("Progress Across Iterations")
            
            # Using matplotlib for more control
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Issues Found', color=color)
            ax1.plot(chart_data["Iteration"], chart_data["Issues Found"], marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            ax2 = ax1.twinx()  # Create a second y-axis
            color = 'tab:red'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(chart_data["Iteration"], chart_data["Accuracy (%)"], marker='s', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            st.pyplot(fig)
    
    def _render_identified_issues(self, review_analysis: Dict[str, Any]):
        """Render identified issues section"""
        identified_problems = review_analysis.get("identified_problems", [])
        
        if not identified_problems:
            st.info("You didn't identify any issues correctly.")
            return
            
        st.subheader(f"Correctly Identified Issues ({len(identified_problems)})")
        
        for i, issue in enumerate(identified_problems, 1):
            st.markdown(
                f"""
                <div style="border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>✓ {i}. {issue}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    def _render_missed_issues(self, review_analysis: Dict[str, Any]):
        """Render missed issues section"""
        missed_problems = review_analysis.get("missed_problems", [])
        
        if not missed_problems:
            st.success("Great job! You identified all the issues.")
            return
            
        st.subheader(f"Issues You Missed ({len(missed_problems)})")
        
        for i, issue in enumerate(missed_problems, 1):
            st.markdown(
                f"""
                <div style="border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>✗ {i}. {issue}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    