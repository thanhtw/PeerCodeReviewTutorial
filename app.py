"""
Java Peer Code Review Training System - LangGraph Version

This module provides a Streamlit web interface for the Java code review training system
using LangGraph for workflow management with a modular UI structure.
Support for both Ollama and Groq LLM providers.
"""

import streamlit as st
import sys
import os
import logging
from dotenv import load_dotenv

# Import CSS utilities
from static.css_utils import load_css

# Configure logging
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import LLM Manager
from llm_manager import LLMManager

# Import LangGraph components
from langgraph_workflow import JavaCodeReviewGraph

# Import modularized UI functions
from ui.main_ui import (
    init_session_state,
    render_sidebar,
    render_llm_logs_tab,
    create_enhanced_tabs
)

# Import UI components
from ui.error_selector import ErrorSelectorUI
from ui.code_display import CodeDisplayUI
from ui.feedback_display import FeedbackDisplayUI
from ui.provider_selector import ProviderSelectorUI
from ui.generate_tab import render_generate_tab
from ui.review_tab import render_review_tab
from ui.feedback_tab import render_feedback_tab

# Load environment variables
load_dotenv(override=True)

# Set page config
st.set_page_config(
    page_title="Java Code Review Trainer",
    page_icon="",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external files
css_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "css")

# Try loading CSS with improved function
loaded_files = load_css(css_directory=css_dir)

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_logs")
if not os.path.exists(log_dir):
     try:
         os.makedirs(log_dir, exist_ok=True)
         logger.info(f"Created log directory: {log_dir}")
     except Exception as e:
         logger.warning(f"Failed to create log directory: {str(e)}")

if not loaded_files:
    # Fallback to inline CSS if loading fails
    logger.warning("Failed to load CSS files, falling back to inline CSS")

def main():
    """Enhanced main application function with provider selection."""
    # Initialize session state
    init_session_state()
    
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    # Initialize provider selector UI
    provider_selector = ProviderSelectorUI(llm_manager)
    
    # Show provider setup modal if needed
    if not provider_selector.render_provider_setup_modal():
        # If provider setup is still needed, don't proceed with the rest of the app
        return
    
    # Initialize workflow after provider is setup
    workflow = JavaCodeReviewGraph(llm_manager)
    
    # Initialize UI components
    error_selector_ui = ErrorSelectorUI()
    code_display_ui = CodeDisplayUI()
    feedback_display_ui = FeedbackDisplayUI()
    
    # Render sidebar with provider status
    render_sidebar(llm_manager, workflow)
    provider_selector.render_provider_status()
    
    # Header with improved styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: rgb(178 185 213); margin-bottom: 5px;">Java Code Review Training System</h1>
        <p style="font-size: 1.1rem; color: #666;">Learn and practice Java code review skills with AI-generated exercises</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display error message if there's an error
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Create enhanced tabs for different steps of the workflow
    tab_labels = [
        "1. Generate Problem", 
        "2. Submit Review", 
        "3. View Feedback",
        "4. LLM Logs"
    ]
    
    # Use the enhanced tabs function
    tabs = create_enhanced_tabs(tab_labels)
    
    # Set the active tab based on session state
    active_tab = st.session_state.active_tab
    
    # Tab content
    with tabs[0]:
        render_generate_tab(workflow, error_selector_ui, code_display_ui)
    
    with tabs[1]:
        render_review_tab(workflow, code_display_ui)
    
    with tabs[2]:
        render_feedback_tab(workflow, feedback_display_ui)
        
    with tabs[3]:  
        render_llm_logs_tab()

if __name__ == "__main__":
    main()