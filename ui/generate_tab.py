"""
Generate Tab UI module for Java Peer Review Training System.

This module provides the functions for rendering the code generation tab
and handling the code generation process.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Callable
from data.json_error_repository import JsonErrorRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_code_problem(workflow, 
                        params: Dict[str, str], 
                        error_selection_mode: str,
                        selected_error_categories: Dict[str, List[str]],
                        selected_specific_errors: List[Dict[str, Any]] = None):
    """Generate a code problem with progress indicator and evaluation visualization."""
    try:
        # Initialize workflow steps
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
        
        # Reset and start tracking steps
        st.session_state.workflow_steps = ["Started code generation process"]
        
        # Initialize state and parameters
        state = st.session_state.workflow_state
        code_length = str(params.get("code_length", "medium"))
        difficulty_level = str(params.get("difficulty_level", "medium"))
        state.code_length = code_length
        state.difficulty_level = difficulty_level
        
        # Verify we have error selections based on mode
        has_selections = False
        if error_selection_mode == "specific" and selected_specific_errors:
            has_selections = len(selected_specific_errors) > 0
            # Update state with specific errors
            state.selected_specific_errors = selected_specific_errors
            # Clear categories for this mode
            state.selected_error_categories = {"build": [], "checkstyle": []}
        elif error_selection_mode == "standard" or error_selection_mode == "advanced":
            build_selected = selected_error_categories.get("build", [])
            checkstyle_selected = selected_error_categories.get("checkstyle", [])
            has_selections = len(build_selected) > 0 or len(checkstyle_selected) > 0
            # Update state with selected categories
            state.selected_error_categories = selected_error_categories
            # Clear specific errors in this mode
            state.selected_specific_errors = []
        
        if not has_selections:
            st.error("No error categories or specific errors selected. Please select at least one error type.")
            return False
        
        # First stage: Generate initial code
        with st.status("Generating initial Java code...", expanded=True) as status:           
            state.current_step = "generate"
            state.evaluation_attempts = 0
            updated_state = workflow.generate_code_node(state)
            st.session_state.workflow_steps.append("Generated initial code")
            
            # Important: Update the session state immediately after code generation
            st.session_state.workflow_state = updated_state
            
            if updated_state.error:
                st.error(f"Error: {updated_state.error}")
                return False
        
        # Second stage: Display the evaluation process
        st.info("Evaluating and improving the code...")
        
        # Create a process visualization using columns and containers instead of expanders
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Code Generation & Evaluation Process")
            
            # Create a progress container
            progress_container = st.container()
            with progress_container:
                # Create a progress bar
                progress_bar = st.progress(0.25)
                st.write("**Step 1:** Initial code generation completed")
                
                # Evaluate the code
                with st.status("Evaluating code quality...", expanded=False):
                    updated_state.current_step = "evaluate"
                    updated_state = workflow.evaluate_code_node(updated_state)
                    # Update session state after each step
                    st.session_state.workflow_state = updated_state
                    st.session_state.workflow_steps.append("Evaluated code for requested errors")
                
                progress_bar.progress(0.5)
                st.write("**Step 2:** Code evaluation completed")
                
                # Show evaluation results
                if hasattr(updated_state, 'evaluation_result') and updated_state.evaluation_result:
                    found = len(updated_state.evaluation_result.get("found_errors", []))
                    missing = len(updated_state.evaluation_result.get("missing_errors", []))
                    total = found + missing
                    if total == 0:
                        total = 1  # Avoid division by zero
                    
                    quality_percentage = (found / total * 100)
                    st.write(f"**Initial quality:** Found {found}/{total} required errors ({quality_percentage:.1f}%)")
                    
                    # Regeneration cycle if needed
                    if missing > 0 and workflow.should_regenerate_or_review(updated_state) == "regenerate_code":
                        st.write("**Step 3:** Improving code quality")
                        
                        attempt = 1
                        max_attempts = getattr(updated_state, 'max_evaluation_attempts', 3)
                        previous_found = found
                        
                        # Loop through regeneration attempts
                        while (attempt < max_attempts and 
                              workflow.should_regenerate_or_review(updated_state) == "regenerate_code"):
                            progress_value = 0.5 + (0.5 * (attempt / max_attempts))
                            progress_bar.progress(progress_value)
                            
                            # Regenerate code
                            with st.status(f"Regenerating code (Attempt {attempt+1})...", expanded=False):
                                updated_state.current_step = "regenerate"
                                updated_state = workflow.regenerate_code_node(updated_state)
                                # Update session state after code regeneration
                                st.session_state.workflow_state = updated_state
                                st.session_state.workflow_steps.append(f"Regenerating code (attempt {attempt+1})")
                            
                            # Re-evaluate code
                            with st.status(f"Re-evaluating code...", expanded=False):
                                updated_state.current_step = "evaluate"
                                updated_state = workflow.evaluate_code_node(updated_state)
                                # Update session state after evaluation
                                st.session_state.workflow_state = updated_state
                                st.session_state.workflow_steps.append(f"Re-evaluated regenerated code")
                            
                            # Show updated results
                            if hasattr(updated_state, 'evaluation_result'):
                                new_found = len(updated_state.evaluation_result.get("found_errors", []))
                                new_missing = len(updated_state.evaluation_result.get("missing_errors", []))
                                
                                st.write(f"**Quality after attempt {attempt+1}:** Found {new_found}/{total} required errors " +
                                      f"({new_found/total*100:.1f}%)")
                                
                                if new_found > previous_found:
                                    st.success(f"✅ Added {new_found - previous_found} new errors in this attempt!")
                                    
                                previous_found = new_found
                            
                            # Increment the attempt counter
                            attempt += 1
                            updated_state.evaluation_attempts = attempt
                            # Make sure to update session state with the new attempt count
                            st.session_state.workflow_state = updated_state
                    
                    # Complete the progress
                    progress_bar.progress(1.0)
                    
                    # Show final outcome
                    if quality_percentage == 100:
                        st.success("✅ All requested errors successfully implemented!")
                    elif quality_percentage >= 80:
                        st.success(f"✅ Good quality code generated with {quality_percentage:.1f}% of requested errors!")
                    else:
                        st.warning(f"⚠️ Code generated with {quality_percentage:.1f}% of requested errors. " +
                                "Some errors could not be implemented but the code is still suitable for review practice.")
                
        with col2:
            # Show statistics in the sidebar
            st.subheader("Generation Stats")
            
            if hasattr(updated_state, 'evaluation_result') and updated_state.evaluation_result:
                found = len(updated_state.evaluation_result.get("found_errors", []))
                missing = len(updated_state.evaluation_result.get("missing_errors", []))
                total = found + missing
                if total > 0:
                    quality_percentage = (found / total * 100)
                    st.metric("Quality", f"{quality_percentage:.1f}%")
                
                st.metric("Errors Found", f"{found}/{total}")
                
                if hasattr(updated_state, 'evaluation_attempts'):
                    st.metric("Generation Attempts", updated_state.evaluation_attempts)
        
        # Update session state with completed process
        updated_state.current_step = "review"
        st.session_state.workflow_state = updated_state
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.error = None
        st.session_state.workflow_steps.append("Code generation process completed successfully")
        
        # Display the generated code
        if hasattr(updated_state, 'code_snippet') and updated_state.code_snippet:
            # Show the generated code in this tab for immediate feedback
            st.subheader("Generated Java Code")
            
            code_to_display = None
            if hasattr(updated_state.code_snippet, 'clean_code') and updated_state.code_snippet.clean_code:
                code_to_display = updated_state.code_snippet.clean_code
            elif hasattr(updated_state.code_snippet, 'code') and updated_state.code_snippet.code:
                code_to_display = updated_state.code_snippet.code
                
            if code_to_display:
                st.code(code_to_display, language="java")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating code problem: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"Error generating code problem: {str(e)}")
        return False

def render_generate_tab(workflow, error_selector_ui, code_display_ui):
    """
    Render the problem generation tab with enhanced workflow visualization.
    
    Args:
        workflow: JavaCodeReviewGraph workflow
        error_selector_ui: ErrorSelectorUI instance
        code_display_ui: CodeDisplayUI instance
    """
    st.subheader("Generate Java Code Review Problem")

    force_regenerate = st.session_state.get("force_regeneration", False)

    if force_regenerate:
        st.session_state.force_regeneration = False
        st.info("Starting a new code review session. Please configure and generate a new problem.")
    

    # Initialize workflow steps if not present
    if not hasattr(st.session_state, 'workflow_steps'):
        st.session_state.workflow_steps = []
    
    # If we already have a code snippet, show the workflow process
    if hasattr(st.session_state, 'workflow_state') and hasattr(st.session_state.workflow_state, 'code_snippet') and st.session_state.workflow_state.code_snippet:
        # First display the workflow progress
        show_workflow_process()
        
        # Then display the generated code
        st.subheader("Generated Java Code:")
        
        # Get known problems from multiple sources to ensure we have data for instructor view
        known_problems = []
        
        # First, try to get problems from evaluation_result['found_errors']
        if (hasattr(st.session_state.workflow_state, 'evaluation_result') and 
            st.session_state.workflow_state.evaluation_result and 
            'found_errors' in st.session_state.workflow_state.evaluation_result):
            found_errors = st.session_state.workflow_state.evaluation_result.get('found_errors', [])
            if found_errors:
                known_problems = found_errors
        
        # If we couldn't get known problems from evaluation, try to get from selected errors
        if not known_problems and hasattr(st.session_state.workflow_state, 'selected_specific_errors'):
            selected_errors = st.session_state.workflow_state.selected_specific_errors
            if selected_errors:
                # Format selected errors to match expected format
                known_problems = [
                    f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                    for error in selected_errors
                ]
        
        # As a last resort, try to extract from raw_errors in code_snippet
        if not known_problems and hasattr(st.session_state.workflow_state.code_snippet, 'raw_errors'):
            raw_errors = st.session_state.workflow_state.code_snippet.raw_errors
            if isinstance(raw_errors, dict):
                for error_type, errors in raw_errors.items():
                    for error in errors:
                        if isinstance(error, dict):
                            error_type_str = error.get('type', error_type).upper()
                            error_name = error.get('name', error.get('error_name', error.get('check_name', 'Unknown')))
                            known_problems.append(f"{error_type_str} - {error_name}")
        
        # Always pass known_problems, the render_code_display function will handle showing
        # the instructor view based on session state and checkbox status
        code_display_ui.render_code_display(
            st.session_state.workflow_state.code_snippet,
            known_problems=known_problems
        )
        
        # Add button to regenerate code
        if st.button("Generate New Problem", type="primary"):
            # Reset the state for new generation
            st.session_state.workflow_state.code_snippet = None
            st.session_state.workflow_state.current_step = "generate"
            st.session_state.workflow_state.evaluation_attempts = 0
            st.session_state.workflow_steps = []
            # Rerun to update UI
            st.rerun()
    else:
        # Display error selection mode
        st.markdown("#### Problem Setup")
        
        selection_mode = error_selector_ui.render_mode_selector()
        
        # Display code generation parameters
        st.markdown("#### Code Parameters")
        params = error_selector_ui.render_code_params()
        
        # Display error selection interface based on mode
        all_categories = workflow.get_all_error_categories()
        
        if selection_mode == "advanced":
            # Advanced mode - select categories
            selected_categories = error_selector_ui.render_category_selection(all_categories)
            specific_errors = []
        else:
            # Specific mode - select specific errors
            specific_errors = error_selector_ui.render_specific_error_selection(workflow.error_repository)
            # Initialize empty categories for this mode
            selected_categories = {"build": [], "checkstyle": []}
        
        # Generate button
        if st.button("Generate Code Problem", type="primary"):
            # Reset workflow steps for a fresh generation
            st.session_state.workflow_steps = ["Started code generation process"]
            
            # Create a placeholder for the status message
            status_placeholder = st.empty()
            status_placeholder.info("Generating Java code with specified errors...")
            
            # Store parameters in state for generation
            if hasattr(st.session_state, 'workflow_state'):
                # Set basic parameters
                st.session_state.workflow_state.code_length = params["code_length"]
                st.session_state.workflow_state.difficulty_level = params["difficulty_level"]
                
                # Set selection based on mode
                if selection_mode == "advanced":
                    st.session_state.workflow_state.selected_error_categories = selected_categories
                    st.session_state.workflow_state.selected_specific_errors = []
                else:
                    st.session_state.workflow_state.selected_error_categories = {"build": [], "checkstyle": []}
                    st.session_state.workflow_state.selected_specific_errors = specific_errors
                
                # Initialize generation state
                st.session_state.workflow_state.current_step = "generate"
                st.session_state.workflow_state.evaluation_attempts = 0
                
                try:
                    # Run the full generate-evaluate-regenerate workflow
                    with st.spinner("Generating and evaluating code..."):
                        # Step 1: Generate initial code
                        state = workflow.generate_code_node(st.session_state.workflow_state)
                        st.session_state.workflow_steps.append("Generated initial code")
                        
                        if state.error:
                            status_placeholder.error(f"Error: {state.error}")
                            return
                        
                        # Step 2: Evaluate code
                        status_placeholder.info("Evaluating generated code for errors...")
                        state = workflow.evaluate_code_node(state)
                        st.session_state.workflow_steps.append("Evaluated code for requested errors")
                        
                        if state.error:
                            status_placeholder.error(f"Error: {state.error}")
                            return
                        
                        # Step 3: Regenerate if needed
                        evaluation_attempts = 1
                        max_attempts = state.max_evaluation_attempts
                        
                        # Loop for regeneration attempts
                        while evaluation_attempts < max_attempts and workflow.should_regenerate_or_review(state) == "regenerate_code":
                            # Update steps and status
                            st.session_state.workflow_steps.append(f"Regenerating code (attempt {evaluation_attempts})")
                            status_placeholder.warning(f"Regenerating code (attempt {evaluation_attempts}/{max_attempts})...")
                            
                            # Regenerate code
                            state.current_step = "regenerate"
                            state = workflow.regenerate_code_node(state)
                            
                            if state.error:
                                status_placeholder.error(f"Error: {state.error}")
                                return
                                
                            # Re-evaluate code
                            st.session_state.workflow_steps.append(f"Re-evaluated regenerated code")
                            status_placeholder.info("Re-evaluating generated code...")
                            state = workflow.evaluate_code_node(state)
                            
                            if state.error:
                                status_placeholder.error(f"Error: {state.error}")
                                return
                                
                            # Increment attempt counter
                            evaluation_attempts += 1
                        
                        # Update state and finalize
                        state.current_step = "review"
                        state.evaluation_attempts = evaluation_attempts
                        st.session_state.workflow_state = state
                        st.session_state.workflow_steps.append("Code generation process completed successfully")
                        
                        # Move to review tab
                        st.session_state.active_tab = 1
                        status_placeholder.success("Code generation complete! Proceeding to review tab...")
                        
                        # Force UI refresh
                        st.rerun()
                        
                except Exception as e:
                    logger.error(f"Error in workflow: {str(e)}", exc_info=True)
                    st.session_state.workflow_steps.append(f"Error in code generation: {str(e)}")
                    status_placeholder.error(f"Error: {str(e)}")
            else:
                # If workflow state doesn't exist yet
                st.error("Workflow state not initialized. Please refresh the page.")

def show_workflow_process():
    """Show a visual representation of the workflow process with improved styling."""
    # Check if we have the necessary state information
    if not hasattr(st.session_state, 'workflow_state'):
        return
    
    state = st.session_state.workflow_state
    
    # Get the current step and evaluation attempts
    current_step = getattr(state, 'current_step', 'generate')
    evaluation_attempts = getattr(state, 'evaluation_attempts', 0)
    max_evaluation_attempts = getattr(state, 'max_evaluation_attempts', 3)
    
    # Create a workflow visualization
    st.subheader("Code Generation Process")
    
    # Define styles exactly as in the screenshot
    # Green color for completed steps
    completed_style = "background-color: #28a745; color: white; text-align: center; padding: 12px; border-radius: 5px; margin: 5px 0; font-weight: 500;"
    # Blue color for active step
    active_style = "background-color: #4c68d7; color: white; text-align: center; padding: 12px; border-radius: 5px; margin: 5px 0; font-weight: 500;"
    # Grey for pending steps
    pending_style = "background-color: #6c757d; color: white; text-align: center; padding: 12px; border-radius: 5px; margin: 5px 0; font-weight: 500;"
    
    # Create columns for the process steps
    cols = st.columns(4)
    
    # Step 1: Generate Code - Always completed if we're showing workflow
    with cols[0]:
        st.markdown(f"<div style='{completed_style}'>1. Generate Code</div>", unsafe_allow_html=True)
    
    # Step 2: Evaluate Code
    with cols[1]:
        evaluate_style = completed_style if current_step in ['evaluate', 'regenerate', 'review'] else pending_style
        st.markdown(f"<div style='{evaluate_style}'>2. Evaluate Code</div>", unsafe_allow_html=True)
    
    # Step 3: Regenerate
    with cols[2]:
        # Set the text for regeneration step
        if evaluation_attempts > 0:
            regenerate_text = f"3. Regenerate ({evaluation_attempts} attempts)"
            regenerate_style = completed_style
        else:
            regenerate_text = "3. Regenerate"
            regenerate_style = pending_style
        
        st.markdown(f"<div style='{regenerate_style}'>{regenerate_text}</div>", unsafe_allow_html=True)
    
    # Step 4: Ready for Review
    with cols[3]:
        review_style = active_style if current_step == 'review' else pending_style
        st.markdown(f"<div style='{review_style}'>4. Ready for Review</div>", unsafe_allow_html=True)
    
    # Show process details in an expander
    with st.expander("Show Process Details", expanded=False):
        if hasattr(st.session_state, 'workflow_steps') and st.session_state.workflow_steps:
            for i, step in enumerate(st.session_state.workflow_steps, 1):
                st.markdown(f"{i}. {step}")
        else:
            st.write("No process details available.")