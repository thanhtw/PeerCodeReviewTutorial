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
        
    # Initialize instructor view flag if not present
    if 'instructor_view' not in st.session_state:
        st.session_state.instructor_view = False
        
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
    """Render the LLM logs tab with detailed log information."""
    st.subheader("LLM Interaction Logs")
    
    if hasattr(st, 'session_state') and 'llm_logger' in st.session_state:
        llm_logger = st.session_state.llm_logger
        logs = llm_logger.get_recent_logs(20)  # Get 20 most recent logs
        
        if logs:
            # Add filter for log types
            log_types = list(set(log.get("type", "unknown") for log in logs))
            log_type_filter = st.multiselect("Filter by log type:", log_types, default=log_types)
            
            # Filter logs
            filtered_logs = [log for log in logs if log.get("type", "unknown") in log_type_filter]
            
            # Display logs
            for log in filtered_logs:
                with st.expander(f"{log.get('timestamp', 'Unknown time')} - {log.get('type', 'Unknown type')}"):
                    # Prompt
                    st.markdown("#### Prompt")
                    st.text_area("", value=log.get("prompt", ""), height=150, key=f"prompt_{log.get('timestamp', '')}")
                    
                    # Response
                    st.markdown("#### Response")
                    st.text_area("", value=log.get("response", ""), height=300, key=f"response_{log.get('timestamp', '')}")
                    
                    # Metadata
                    with st.expander("Metadata"):
                        st.json(log.get("metadata", {}))
        else:
            st.info("No logs recorded yet. Generate code or submit reviews to see logs.")
    else:
        st.info("LLM logger not initialized.")
    
    # Add clear logs button
    if st.button("Clear Logs"):
        if hasattr(st, 'session_state') and 'llm_logger' in st.session_state:
            st.session_state.llm_logger.clear_logs()
            st.success("Logs cleared.")
            st.rerun()

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

def render_status_sidebar(llm_manager):
    """
    Render status information in the sidebar.
    
    Args:
        llm_manager: LLMManager instance for accessing status information
    """
    st.sidebar.title("System Status")
    
    # Display provider info
    provider = llm_manager.provider
    st.sidebar.markdown(f"**Provider:** {provider.upper()}")
    
    # Check connection status
    if provider == "ollama":
        connection_status, message = llm_manager.check_ollama_connection()
        if connection_status:
            st.sidebar.success("✅ Connected to Ollama")
            
            # Get GPU info - only relevant for Ollama
            gpu_info = llm_manager.refresh_gpu_info()
            
            # Handle None gpu_info
            if gpu_info is not None:
                has_gpu = gpu_info.get("has_gpu", False)
                
                if has_gpu:
                    gpu_name = gpu_info.get("gpu_name", "GPU")
                    memory_total = gpu_info.get("formatted_total", "Unknown")
                    memory_used = gpu_info.get("formatted_used", "Unknown")
                    
                    st.sidebar.markdown("**GPU Status:**")
                    st.sidebar.markdown(f"🚀 {gpu_name}")
                    st.sidebar.markdown(f"Memory: {memory_used}/{memory_total}")
                else:
                    st.sidebar.warning("⚠️ No GPU detected - using CPU only")
            else:
                st.sidebar.warning("⚠️ GPU info not available - using CPU only")
        else:
            st.sidebar.error(f"❌ Ollama connection error: {message}")
    elif provider == "groq":
        connection_status, message = llm_manager.check_groq_connection()
        if connection_status:
            st.sidebar.success("✅ Connected to Groq API")
            st.sidebar.info("☁️ Using cloud-based models (GPU status not applicable)")
            
            # Display API key info (masked)
            api_key = llm_manager.groq_api_key
            if api_key:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "Set"
                st.sidebar.markdown(f"**API Key:** {masked_key}")
        else:
            st.sidebar.error(f"❌ Groq API connection error: {message}")
    else:
        st.sidebar.warning(f"⚠️ Unknown provider: {provider}")
    
    # Show active models
    active_models = llm_manager.get_active_models()
    if active_models:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Active Models:**")
        
        for role, model_info in active_models.items():
            model_name = model_info.get("name", "Unknown")
            st.sidebar.markdown(f"- **{role.title()}**: {model_name}")
            
    # Add refresh button
    if st.sidebar.button("Refresh Status"):
        st.rerun()

def render_settings_sidebar(workflow):
    """Render the settings sidebar tab"""
    st.header("Review Settings")
    
    # Create a settings card
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Review iteration settings
    max_iterations = st.slider(
        "Maximum Review Attempts",
        min_value=1,
        max_value=5,
        value=st.session_state.workflow_state.max_iterations,
        help="Maximum attempts before final evaluation"
    )
    
    # Update max iterations in workflow state
    if max_iterations != st.session_state.workflow_state.max_iterations:
        st.session_state.workflow_state.max_iterations = max_iterations
    
    # Minimum identified percentage for sufficient review
    min_identified_percentage = st.slider(
        "Required Accuracy",
        min_value=30,
        max_value=90,
        value=60,
        step=5,
        help="Minimum % of issues to identify for a sufficient review"
    )
    
    # Update student response evaluator
    if hasattr(workflow, 'evaluator') and hasattr(workflow.evaluator, 'min_identified_percentage'):
        workflow.evaluator.min_identified_percentage = min_identified_percentage
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display settings
    st.subheader("Display Settings")
    
    # Create display settings card
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Toggle for instructor view
    instructor_view = st.checkbox(
        "Enable Instructor View",
        value=False,
        help="Show known errors (for instructors)"
    )
    
    if "instructor_view" not in st.session_state or instructor_view != st.session_state.instructor_view:
        st.session_state.instructor_view = instructor_view
    
    st.markdown('</div>', unsafe_allow_html=True)

def check_ollama_status(llm_manager: LLMManager) -> Dict[str, bool]:
    """
    Check the status of Ollama and required models.
    """
    # Check Ollama connection
    connection_status, _ = llm_manager.check_ollama_connection()
    
    # Check if default model is available
    default_model_available = False
    if connection_status:
        default_model = llm_manager.default_model
        default_model_available = llm_manager.check_model_availability(default_model)
    
    # Check if all role-specific models are configured in environment
    required_models = ["GENERATIVE_MODEL", "REVIEW_MODEL", "SUMMARY_MODEL", "COMPARE_MODEL"]
    all_models_configured = all(os.getenv(model) for model in required_models)
    
    # Check GPU status
    gpu_info = llm_manager.check_gpu_availability()
    has_gpu = gpu_info.get("has_gpu", False)
    
    return {
        "ollama_running": connection_status,
        "default_model_available": default_model_available,
        "all_models_configured": all_models_configured,
        "gpu_available": has_gpu
    }

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
    
    # For debugging
    logger.debug(f"Current active tab: {current_tab}")
    
    # Force-select the active tab
    if current_tab != 0:
        # This doesn't actually change the UI, but it helps with logic elsewhere
        st.session_state.active_tab = current_tab
    
    return tabs

def show_workflow_progress():
    """Show a visual indicator of the workflow progress."""
    if 'current_step' not in st.session_state:
        return
    
    current_step = st.session_state.current_step
    evaluation_attempts = st.session_state.get('evaluation_attempts', 0)
    max_evaluation_attempts = st.session_state.get('max_evaluation_attempts', 3)
    
    # Create a progress bar based on the current step
    st.markdown("### Workflow Progress")
    
    steps = ["generate", "evaluate", "regenerate", "review", "analyze", "summarize"]    
    
    try:
        # Find the index of the current step
        current_index = steps.index(current_step)
    except ValueError:
        current_index = 0
    
    # Calculate progress percentage
    progress = (current_index + 1) / len(steps)
    
    # Show progress bar
    st.progress(progress)
    
    # Show current step with detailed status
    if current_step == "generate":
        st.info("🔍 **Generate Code**: Creating Java code with specified errors")
    elif current_step == "evaluate":
        st.info("⚙️ **Evaluate Code**: Checking if code contains required errors")
    elif current_step == "regenerate":
        st.info(f"🔄 **Regenerate Code**: Attempt {evaluation_attempts}/{max_evaluation_attempts} to fix missing errors")
    elif current_step == "review":
        st.success("✅ **Review Ready**: Code generation complete, ready for review")
    elif current_step == "analyze":
        st.info("📊 **Analyze Review**: Evaluating your review submission")
    elif current_step == "summarize":
        st.success("🎓 **Complete**: Review process completed, feedback ready")
    else:
        st.info(f"Current step: {current_step}")
    
    # Show detailed workflow steps if available
    if 'workflow_steps' in st.session_state and st.session_state.workflow_steps:
        with st.expander("Workflow History"):
            for i, step in enumerate(st.session_state.workflow_steps):
                st.markdown(f"{i+1}. {step}")

def render_sidebar(llm_manager, workflow):
    """
    Render the sidebar with application info, model status, and workflow visualization.
    
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