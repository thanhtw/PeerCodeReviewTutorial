"""
Main UI components for Java Peer Review Training System.

This module provides the primary UI functions for the Streamlit interface,
including tab rendering, LLM logs, and sidebar components.
"""

import streamlit as st
import os
import logging
import re
import io
import zipfile
import base64
import time
from typing import Dict, List, Any, Optional, Callable

from utils.llm_logger import LLMInteractionLogger
from llm_manager import LLMManager
from state_schema import WorkflowState

# Configure logging
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state with default values."""
    # Initialize workflow state if not present
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = WorkflowState()
    
    # Initialize active tab if not present
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Initialize error state if not present
    if 'error' not in st.session_state:
        st.session_state.error = None
    
    # Initialize workflow steps if not present
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = []
        
    # Initialize sidebar state if not present
    if 'sidebar_tab' not in st.session_state:
        st.session_state.sidebar_tab = "Status"
        
    # Initialize current step if needed - directly in workflow_state
    if hasattr(st.session_state, 'workflow_state') and not hasattr(st.session_state.workflow_state, 'current_step'):
        st.session_state.workflow_state.current_step = "generate"
        
    # Initialize evaluation attempts if needed - directly in workflow_state
    if hasattr(st.session_state, 'workflow_state') and not hasattr(st.session_state.workflow_state, 'evaluation_attempts'):
        st.session_state.workflow_state.evaluation_attempts = 0
        
    # Initialize max evaluation attempts if needed - directly in workflow_state
    if hasattr(st.session_state, 'workflow_state') and not hasattr(st.session_state.workflow_state, 'max_evaluation_attempts'):
        st.session_state.workflow_state.max_evaluation_attempts = 3
    
    # Initialize LLM logger if not present
    if 'llm_logger' not in st.session_state:
        st.session_state.llm_logger = LLMInteractionLogger()
        
    # Ensure code_snippet is never accessed directly, but through workflow_state
    if 'code_snippet' in st.session_state:
        # Transfer any existing code_snippet to workflow_state and remove direct reference
        if 'workflow_state' in st.session_state and not hasattr(st.session_state.workflow_state, 'code_snippet'):
            st.session_state.workflow_state.code_snippet = st.session_state.code_snippet
        # Remove the direct reference to avoid confusion
        del st.session_state.code_snippet

def render_llm_logs_tab():
    """Render the LLM logs tab with detailed log information and file browsing capabilities."""
    st.subheader("LLM Interaction Logs")
    
    if hasattr(st, 'session_state') and 'llm_logger' in st.session_state:
        llm_logger = st.session_state.llm_logger
        
        # Add refresh button at the top
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Refresh Logs", key="refresh_logs_btn"):
                st.rerun()
        
        with col1:
            # Make the log count configurable
            log_count = st.slider("Number of logs to display", min_value=5, max_value=30, value=10, step=5)
        
        # Get logs (will now include both in-memory and disk logs)
        logs = llm_logger.get_recent_logs(log_count)
        
        if logs:
            # Add filter for log types
            log_types = sorted(list(set(log.get("type", "unknown") for log in logs)))
            log_type_filter = st.multiselect("Filter by log type:", log_types, default=log_types)
            
            # Date filter
            timestamps = [log.get("timestamp", "") for log in logs if "timestamp" in log]
            if timestamps:
                # Extract dates from timestamps
                dates = sorted(set(ts.split("T")[0] for ts in timestamps if "T" in ts))
                if dates:
                    selected_dates = st.multiselect("Filter by date:", dates, default=dates)
                    # Apply date filter
                    logs = [log for log in logs if "timestamp" in log and log["timestamp"].split("T")[0] in selected_dates]
            
            # Filter logs by type
            filtered_logs = [log for log in logs if log.get("type", "unknown") in log_type_filter]
            
            if filtered_logs:
                st.success(f"Displaying {len(filtered_logs)} recent logs. Newest logs appear first.")
                
                # Display logs with improved UI
                for log in filtered_logs:
                    # Format timestamp for display
                    timestamp = log.get("timestamp", "Unknown time")
                    if "T" in timestamp:
                        date, time = timestamp.split("T")
                        time = time.split(".")[0] if "." in time else time  # Remove milliseconds
                        display_time = f"{date} {time}"
                    else:
                        display_time = timestamp
                    
                    # Create expander title with log type and timestamp
                    log_type = log.get("type", "Unknown type").replace("_", " ").title()
                    expander_title = f"{log_type} - {display_time}"
                    
                    with st.expander(expander_title):
                        # Create tabs for different parts of the log
                        log_tabs = st.tabs(["Prompt", "Response", "Metadata"])
                        
                        # Prompt tab
                        with log_tabs[0]:
                            st.text_area(
                                "Prompt sent to LLM:", 
                                value=log.get("prompt", ""), 
                                height=250,
                                key=f"prompt_{log.get('timestamp', '')}",
                                disabled=True
                            )
                        
                        # Response tab
                        with log_tabs[1]:
                            response = log.get("response", "")
                            if response:
                                # Check if response contains code blocks and highlight them
                                if "```" in response:
                                    parts = response.split("```")
                                    for i, part in enumerate(parts):
                                        if i % 2 == 0:  # Regular text
                                            if part.strip():
                                                st.markdown(part)
                                        else:  # Code block
                                            language = ""
                                            if part.strip() and "\n" in part:
                                                language_line, code = part.split("\n", 1)
                                                if language_line.strip():
                                                    language = language_line.strip()
                                                    part = code
                                            
                                            # Display code with syntax highlighting if language is specified
                                            if language:
                                                st.code(part, language=language)
                                            else:
                                                st.code(part)
                                else:
                                    # Show as plain text
                                    st.text_area(
                                        "Response:", 
                                        value=response, 
                                        height=300,
                                        key=f"response_{log.get('timestamp', '')}",
                                        disabled=True
                                    )
                            else:
                                st.info("No response available")
                        
                        # Metadata tab
                        with log_tabs[2]:
                            metadata = log.get("metadata", {})
                            if metadata:
                                st.json(metadata)
                            else:
                                st.info("No metadata available")
            else:
                st.info("No logs match the selected filters.")
        else:
            st.info("No logs found. Generate code or submit reviews to create log entries.")
            
            # Add helper information about log location
            st.markdown("""
            ### Log Information
            
            Log files are stored in the `llm_logs` directory, with subdirectories for each interaction type:
            
            - code_generation
            - code_regeneration
            - code_evaluation
            - review_analysis
            - summary_generation
            
            Each log is stored as both a `.json` file (for programmatic use) and a `.txt` file (for easier reading).
            """)
    else:
        st.info("LLM logger not initialized.")
    
    # Add clear logs button with confirmation
    st.markdown("---")
    if st.button("Clear Logs"):
        st.warning("This will remove in-memory logs. Log files on disk will be preserved.")
        confirm_key = "confirm_clear_logs"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False
        
        if st.session_state[confirm_key] or st.button("Confirm Clear Logs", key="confirm_clear_btn"):
            if hasattr(st, 'session_state') and 'llm_logger' in st.session_state:
                st.session_state.llm_logger.clear_logs()
                st.session_state[confirm_key] = False
                st.success("Logs cleared.")
                st.rerun()
            else:
                st.error("LLM logger not initialized.")

def render_sidebar(llm_manager, workflow):
    """
    Render the sidebar with application info and model status.
    
    Args:
        llm_manager: LLMManager instance
        workflow: JavaCodeReviewGraph workflow
    """
    with st.sidebar:
        st.title("Java Review Trainer")
        
        # Application info
        st.subheader("About")
        st.markdown("""
        This application helps you learn and practice Java code review skills
        by generating code with intentional errors for you to identify.
        """)
        
        # Add separator
        st.markdown("---")
        
        # LLM Provider info
        st.subheader("LLM Provider")
        provider = llm_manager.provider.capitalize()
        
        if provider == "Ollama":
            connection_status, message = llm_manager.check_ollama_connection()
            status = "✅ Connected" if connection_status else "❌ Disconnected"
            st.markdown(f"**Provider:** {provider}  \n**Status:** {status}")
            
            if not connection_status:
                st.error(f"Error: {message}")
                st.markdown("""
                Make sure Ollama is running:
                ```bash
                ollama serve
                ```
                """)
        elif provider == "Groq":
            connection_status, message = llm_manager.check_groq_connection()
            status = "✅ Connected" if connection_status else "❌ Disconnected"
            st.markdown(f"**Provider:** {provider}  \n**Status:** {status}")
            
            if not connection_status:
                st.error(f"Error: {message}")
                st.markdown("""
                Check your Groq API key in the .env file:
                ```
                GROQ_API_KEY=your_api_key
                ```
                """)
        
        # Add separator
        st.markdown("---")
        
        # Reset button
        if st.button("Reset Application", use_container_width=True):
            # Reset the session state
            for key in list(st.session_state.keys()):
                # Keep provider selection and error categories
                if key not in ["provider", "selected_error_categories"]:
                    del st.session_state[key]
            
            # Reinitialize session state
            init_session_state()
            
            # Set active tab to generate
            st.session_state.active_tab = 0
            
            # Rerun app
            st.rerun()

def create_enhanced_tabs(labels: List[str]):
    """
    Create enhanced UI tabs with tab switching capability.
    
    Args:
        labels: List of tab labels
        
    Returns:
        List of tab objects
    """
    # Create tabs with enhanced styling
    tabs = st.tabs(labels)
    
    # Handle tab switching based on session state
    current_tab = st.session_state.active_tab
    
    # Check if we need to block access to feedback tab
    if hasattr(st.session_state, 'workflow_state'):
        state = st.session_state.workflow_state
        # Determine if review is completed
        review_completed = False
        
        # Check if max iterations reached or review is sufficient
        if hasattr(state, 'current_iteration') and hasattr(state, 'max_iterations'):
            if state.current_iteration > state.max_iterations:
                review_completed = True
            elif hasattr(state, 'review_sufficient') and state.review_sufficient:
                review_completed = True
        
        # Block access to feedback tab (index 2) if review not completed
        if current_tab == 2 and not review_completed:
            st.warning("Please complete all review attempts before accessing feedback.")
            # Reset to review tab
            st.session_state.active_tab = 1
            current_tab = 1
    
    # Force-select the active tab
    if current_tab != 0:
        # This doesn't actually change the UI, but it helps with logic elsewhere
        st.session_state.active_tab = current_tab
    
    return tabs
