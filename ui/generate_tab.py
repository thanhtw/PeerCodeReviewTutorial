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
        # Initialize state and parameters
        state = st.session_state.workflow_state
        code_length = str(params.get("code_length", "medium"))
        difficulty_level = str(params.get("difficulty_level", "medium"))
        state.code_length = code_length
        state.difficulty_level = difficulty_level
        
        # Verify we have error selections
        has_selections = False
        if error_selection_mode == "specific" and selected_specific_errors:
            has_selections = len(selected_specific_errors) > 0
        elif error_selection_mode == "standard" or error_selection_mode == "advanced":
            build_selected = selected_error_categories.get("build", [])
            checkstyle_selected = selected_error_categories.get("checkstyle", [])
            has_selections = len(build_selected) > 0 or len(checkstyle_selected) > 0
        
        if not has_selections:
            st.error("No error categories or specific errors selected. Please select at least one error type.")
            return False
        
        # Update the state with selected error categories
        state.selected_error_categories = selected_error_categories
        
        # First stage: Generate initial code
        with st.status("Generating initial Java code...", expanded=True) as status:           
            updated_state = workflow.generate_code_node(state)
            
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
                    updated_state = workflow.evaluate_code_node(updated_state)
                
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
                    if missing > 0 and updated_state.current_step == "regenerate":
                        st.write("**Step 3:** Improving code quality")
                        
                        attempt = 1
                        max_attempts = getattr(updated_state, 'max_evaluation_attempts', 3)
                        previous_found = found
                        
                        # Loop through regeneration attempts
                        while (getattr(updated_state, 'current_step', None) == "regenerate" and 
                              attempt < max_attempts):
                            progress_value = 0.5 + (0.5 * (attempt / max_attempts))
                            progress_bar.progress(progress_value)
                            
                            # Regenerate code
                            with st.status(f"Regenerating code (Attempt {attempt+1})...", expanded=False):
                                updated_state = workflow.regenerate_code_node(updated_state)
                            
                            # Re-evaluate code
                            with st.status(f"Re-evaluating code...", expanded=False):
                                updated_state = workflow.evaluate_code_node(updated_state)
                            
                            # Show updated results
                            if hasattr(updated_state, 'evaluation_result'):
                                new_found = len(updated_state.evaluation_result.get("found_errors", []))
                                new_missing = len(updated_state.evaluation_result.get("missing_errors", []))
                                
                                st.write(f"**Quality after attempt {attempt+1}:** Found {new_found}/{total} required errors " +
                                      f"({new_found/total*100:.1f}%)")
                                
                                if new_found > previous_found:
                                    st.success(f"✅ Added {new_found - previous_found} new errors in this attempt!")
                                    
                                previous_found = new_found
                            
                            attempt += 1
                    
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
        
        # Update session state
        st.session_state.workflow_state = updated_state
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.error = None
        
        # Debug output
        if hasattr(updated_state, 'code_snippet') and updated_state.code_snippet:
            # Also show the generated code in this tab for immediate feedback
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
    
    # If we already have a code snippet, show the workflow process
    if hasattr(st.session_state, 'code_snippet') and st.session_state.code_snippet:
        # First display the workflow progress
        show_workflow_process()
        
        # Then display the generated code
        st.subheader("Generated Java Code:")
        code_display_ui.render_code_display(
            st.session_state.code_snippet,
            # Show known problems only if we need the instructor view
            known_problems=st.session_state.code_snippet.known_problems
        )
        
        # Add button to regenerate code
        if st.button("Generate New Problem", type="primary"):
            # Clear the code snippet
            st.session_state.code_snippet = None
            st.session_state.current_step = "generate"
            st.session_state.evaluation_attempts = 0
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
        
        # Display error category selection based on mode
        all_categories = workflow.get_all_error_categories()
        
        if selection_mode == "advanced":
            # Advanced mode - select categories
            selected_categories = error_selector_ui.render_category_selection(all_categories)
        else:
            # Specific mode - select specific errors
            specific_errors = error_selector_ui.render_specific_error_selection(workflow.error_repository)
        
        # Generate button
        if st.button("Generate Code Problem", type="primary"):
            with st.spinner("Generating Java code problem..."):
                # Store code parameters in session state
                st.session_state.code_length = params["code_length"]
                st.session_state.difficulty_level = params["difficulty_level"]
                
                # Add the current step to workflow steps history
                if 'workflow_steps' not in st.session_state:
                    st.session_state.workflow_steps = []
                
                st.session_state.workflow_steps.append("Started code generation process")
                st.session_state.current_step = "generate"
                
                # Create a placeholder for the status message
                status_placeholder = st.empty()
                
                # Update the status
                status_placeholder.info("Generating Java code with specified errors...")
                
                # Generate code with workflow
                try:
                    # Initialize workflow state
                    from state_schema import WorkflowState
                    
                    # Create initial state
                    state = WorkflowState(
                        code_length=params["code_length"],
                        difficulty_level=params["difficulty_level"],
                        selected_error_categories=selected_categories if selection_mode == "advanced" else {},
                        current_step="generate",
                        evaluation_attempts=0,
                        max_evaluation_attempts=3
                    )
                    
                    # Set specific errors if in specific mode
                    if selection_mode == "specific" and specific_errors:
                        state.selected_specific_errors = specific_errors
                    
                    # Use workflow to generate code
                    state = workflow.generate_code_node(state)
                    
                    # Update status to evaluation
                    status_placeholder.info("Evaluating generated code for errors...")
                    st.session_state.workflow_steps.append("Generated initial code")
                    st.session_state.current_step = "evaluate"
                    
                    # Evaluate the code
                    state = workflow.evaluate_code_node(state)
                    st.session_state.workflow_steps.append("Evaluated code for requested errors")
                    
                    # Check if we need to regenerate (loop until valid or max attempts)
                    evaluation_attempts = 1
                    max_attempts = 3
                    
                    while workflow.should_regenerate_or_review(state) == "regenerate_code" and evaluation_attempts < max_attempts:
                        # Update status to regeneration
                        st.session_state.current_step = "regenerate"
                        status_placeholder.warning(f"Regenerating code (attempt {evaluation_attempts}/{max_attempts})...")
                        st.session_state.workflow_steps.append(f"Regenerating code (attempt {evaluation_attempts})")
                        
                        # Regenerate code
                        state = workflow.regenerate_code_node(state)
                        
                        # Update status to evaluation
                        st.session_state.current_step = "evaluate"
                        status_placeholder.info("Re-evaluating generated code...")
                        st.session_state.workflow_steps.append(f"Re-evaluated regenerated code")
                        
                        # Evaluate the code again
                        state = workflow.evaluate_code_node(state)
                        
                        # Increment attempt counter
                        evaluation_attempts += 1
                    
                    # Update status to review
                    st.session_state.current_step = "review"
                    status_placeholder.success("Code generation complete!")
                    st.session_state.workflow_steps.append("Code generation process completed successfully")
                    
                    # Update session state with workflow state
                    for key, value in state.__dict__.items():
                        setattr(st.session_state, key, value)
                    
                    # Store evaluation attempts
                    st.session_state.evaluation_attempts = evaluation_attempts
                    
                    # Rerun to update UI
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Error generating code: {str(e)}")
                    st.session_state.workflow_steps.append(f"Error in code generation: {str(e)}")
                    status_placeholder.error(f"Error: {str(e)}")

def show_workflow_process():
    """Show a visual representation of the workflow process."""
    if 'current_step' not in st.session_state:
        return
    
    current_step = st.session_state.current_step
    evaluation_attempts = st.session_state.get('evaluation_attempts', 0)
    max_evaluation_attempts = st.session_state.get('max_evaluation_attempts', 3)
    
    # Create a workflow visualization
    st.markdown("### Code Generation Process")
    
    # Create columns for each step
    cols = st.columns(4)
    
    # Define step styles
    active_style = "background-color: #4c68d7; color: white; padding: 8px; border-radius: 5px; text-align: center; margin: 3px;"
    inactive_style = "background-color: #e9ecef; color: #6c757d; padding: 8px; border-radius: 5px; text-align: center; margin: 3px;"
    completed_style = "background-color: #28a745; color: white; padding: 8px; border-radius: 5px; text-align: center; margin: 3px;"
    
    # Step 1: Generate Code
    generate_style = completed_style if current_step != "generate" else active_style
    cols[0].markdown(f"<div style='{generate_style}'>1. Generate Code</div>", unsafe_allow_html=True)
    
    # Step 2: Evaluate Code
    evaluate_style = active_style if current_step == "evaluate" else inactive_style
    evaluate_style = completed_style if current_step not in ["generate", "evaluate"] else evaluate_style
    cols[1].markdown(f"<div style='{evaluate_style}'>2. Evaluate Code</div>", unsafe_allow_html=True)
    
    # Step 3: Regenerate (if needed)
    regenerate_style = active_style if current_step == "regenerate" else inactive_style
    regenerate_style = completed_style if current_step not in ["generate", "evaluate", "regenerate"] and evaluation_attempts > 1 else regenerate_style
    if evaluation_attempts > 1:
        cols[2].markdown(f"<div style='{regenerate_style}'>3. Regenerate ({evaluation_attempts-1} attempts)</div>", unsafe_allow_html=True)
    else:
        cols[2].markdown(f"<div style='{inactive_style}'>3. Regenerate (skipped)</div>", unsafe_allow_html=True)
    
    # Step 4: Ready for Review
    review_style = active_style if current_step == "review" else inactive_style
    review_style = completed_style if current_step not in ["generate", "evaluate", "regenerate", "review"] else review_style
    cols[3].markdown(f"<div style='{review_style}'>4. Ready for Review</div>", unsafe_allow_html=True)
    
    # Add details about the process
    if 'workflow_steps' in st.session_state and st.session_state.workflow_steps:
        with st.expander("Show Process Details"):
            for i, step in enumerate(st.session_state.workflow_steps):
                st.markdown(f"{i+1}. {step}")
    
    # Add details about evaluation results
    if 'evaluation_result' in st.session_state and st.session_state.evaluation_result:
        with st.expander("Show Evaluation Results"):
            result = st.session_state.evaluation_result
            
            # Show found errors
            if "found_errors" in result:
                st.markdown("#### ✅ Implemented Errors")
                for error in result["found_errors"]:
                    st.markdown(f"**{error.get('error_type', '')}-{error.get('error_name', '')}**  \n"
                              f"Line {error.get('line_number', 'Unknown')}: `{error.get('code_segment', '')}`  \n"
                              f"{error.get('explanation', '')}")
            
            # Show missing errors
            if "missing_errors" in result:
                st.markdown("#### ❌ Missing Errors")
                for error in result["missing_errors"]:
                    st.markdown(f"**{error.get('error_type', '')}-{error.get('error_name', '')}**:  \n"
                              f"{error.get('explanation', '')}")
            
            # Show overall validity
            valid = result.get("valid", False)
            if valid:
                st.success("✅ All required errors successfully implemented")
            else:
                st.warning("⚠️ Some required errors are not properly implemented")