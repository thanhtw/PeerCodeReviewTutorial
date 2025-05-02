"""
Error Selector UI module for Java Peer Review Training System.

This module provides the ErrorSelectorUI class for selecting Java error categories
to include in the generated code problems.
"""

import streamlit as st
import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorSelectorUI:
    """
    UI Component for selecting Java error categories.
    
    This class handles displaying and selecting Java error categories
    from both build errors and code quality errors.
    """
    
    def __init__(self):
        """Initialize the ErrorSelectorUI component with empty selections."""
        # Track selected categories - initialize with empty collections if not in session state
        if "selected_error_categories" not in st.session_state:
            st.session_state.selected_error_categories = {
                "build": [],
                "checkstyle": []
            }
        elif not isinstance(st.session_state.selected_error_categories, dict):
            # Fix if it's not a proper dictionary
            st.session_state.selected_error_categories = {
                "build": [],
                "checkstyle": []
            }
        elif "build" not in st.session_state.selected_error_categories or "checkstyle" not in st.session_state.selected_error_categories:
            # Make sure both build and checkstyle keys exist
            if "build" not in st.session_state.selected_error_categories:
                st.session_state.selected_error_categories["build"] = []
            if "checkstyle" not in st.session_state.selected_error_categories:
                st.session_state.selected_error_categories["checkstyle"] = []
        
        # Track error selection mode
        if "error_selection_mode" not in st.session_state:
            st.session_state.error_selection_mode = "advanced"
        
        # Track expanded categories
        if "expanded_categories" not in st.session_state:
            st.session_state.expanded_categories = {}
            
        # Track selected specific errors - initialize as empty list
        if "selected_specific_errors" not in st.session_state:
            st.session_state.selected_specific_errors = []
    
    def render_category_selection(self, all_categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Render the error category selection UI for advanced mode with enhanced debugging.
        
        Args:
            all_categories: Dictionary with 'build' and 'checkstyle' categories
            
        Returns:
            Dictionary with selected categories
        """
        st.subheader("Select Specific Error Categories")
        
        # Add help text explaining how this mode works
        st.info("""
        **Advanced Mode**: Select specific error categories to include in the generated code. 
        When you select a category, the system will randomly choose errors from that category 
        to include in the generated code.
        """)
        
        build_categories = all_categories.get("build", [])
        checkstyle_categories = all_categories.get("checkstyle", [])
        
        # Ensure the session state structure is correct
        if "selected_error_categories" not in st.session_state:
            st.session_state.selected_error_categories = {"build": [], "checkstyle": []}
        if "build" not in st.session_state.selected_error_categories:
            st.session_state.selected_error_categories["build"] = []
        if "checkstyle" not in st.session_state.selected_error_categories:
            st.session_state.selected_error_categories["checkstyle"] = []
        
        # Build errors section
        st.markdown("<div class='section-card'>BUILD ISSUEs</div>", unsafe_allow_html=True)
        
        # Create a multi-column layout for build errors
        build_cols = st.columns(2)
        
        # Get the current selection state from session
        current_build_selections = st.session_state.selected_error_categories.get("build", [])
        
        # Split the build categories into two columns
        half_length = len(build_categories) // 2
        for i, col in enumerate(build_cols):
            start_idx = i * half_length
            end_idx = start_idx + half_length if i == 0 else len(build_categories)
            
            with col:
                for category in build_categories[start_idx:end_idx]:
                    # Create a unique key for this category
                    category_key = f"build_{category}"
                    
                    # Check if category is already selected from session state
                    is_selected = category in current_build_selections
                    
                    # Create the checkbox
                    selected = st.checkbox(
                        category,
                        key=category_key,
                        value=is_selected
                    )
                    
                    # Update selection state based on checkbox
                    if selected:
                        if category not in current_build_selections:
                            current_build_selections.append(category)
                    else:
                        if category in current_build_selections:
                            current_build_selections.remove(category)
        
        # Update build selections in session state
        st.session_state.selected_error_categories["build"] = current_build_selections
        
        # Checkstyle errors section
        st.markdown("<div class='section-card'>CODE QUALITY ISSUES</div>", unsafe_allow_html=True)
        
        # Create a multi-column layout for checkstyle errors
        checkstyle_cols = st.columns(2)
        
        # Get the current selection state from session
        current_checkstyle_selections = st.session_state.selected_error_categories.get("checkstyle", [])
        
        # Split the checkstyle categories into two columns
        half_length = len(checkstyle_categories) // 2
        for i, col in enumerate(checkstyle_cols):
            start_idx = i * half_length
            end_idx = start_idx + half_length if i == 0 else len(checkstyle_categories)
            
            with col:
                for category in checkstyle_categories[start_idx:end_idx]:
                    # Create a unique key for this category
                    category_key = f"checkstyle_{category}"
                    
                    # Check if category is already selected from session state
                    is_selected = category in current_checkstyle_selections
                    
                    # Create the checkbox
                    selected = st.checkbox(
                        category,
                        key=category_key,
                        value=is_selected
                    )
                    
                    # Update selection state based on checkbox
                    if selected:
                        if category not in current_checkstyle_selections:
                            current_checkstyle_selections.append(category)
                    else:
                        if category in current_checkstyle_selections:
                            current_checkstyle_selections.remove(category)
        
        # Update checkstyle selections in session state
        st.session_state.selected_error_categories["checkstyle"] = current_checkstyle_selections
        
        # Selection summary
        build_selected = st.session_state.selected_error_categories.get("build", [])
        checkstyle_selected = st.session_state.selected_error_categories.get("checkstyle", [])
        
        st.write("### Selected Categories")
        if not build_selected and not checkstyle_selected:
            st.warning("No categories selected. You must select at least one category to generate code.")
        
        if build_selected:
            st.write("Build ISSUES Categories:")
            for category in build_selected:
                st.markdown(f"<div class='error-category'>{category}</div>", unsafe_allow_html=True)
        
        if checkstyle_selected:
            st.write("CODE QUALITY ISSUE Categories:")
            for category in checkstyle_selected:
                st.markdown(f"<div class='error-category'>{category}</div>", unsafe_allow_html=True)
        
        # Enhanced debug output for advanced mode
        print("\n========== ADVANCED MODE SELECTION ==========")
        print(f"Build Categories: {build_selected}")
        print(f"Checkstyle Categories: {checkstyle_selected}")
        
        # If any categories are selected, show potential errors in each category
        if build_selected or checkstyle_selected:
            print("\n--- POTENTIAL ERRORS BY CATEGORY ---")
            
            # For build categories
            for category in build_selected:
                print(f"Build Category: {category}")
                # Get sample errors from this category if available
                try:
                    from data.json_error_repository import JsonErrorRepository
                    repo = JsonErrorRepository()
                    errors = repo.get_category_errors("build", category)
                    if errors:
                        print(f"  Sample errors (max 3):")
                        for i, error in enumerate(errors[:3]):
                            print(f"    {i+1}. {error.get('error_name', 'Unknown')}: {error.get('description', '')[:50]}..." 
                                if len(error.get('description', '')) > 50 else error.get('description', ''))
                except Exception as e:
                    print(f"  Error retrieving category info: {str(e)}")
            
            # For checkstyle categories
            for category in checkstyle_selected:
                print(f"Checkstyle Category: {category}")
                # Get sample errors from this category if available
                try:
                    from data.json_error_repository import JsonErrorRepository
                    repo = JsonErrorRepository()
                    errors = repo.get_category_errors("checkstyle", category)
                    if errors:
                        print(f"  Sample errors (max 3):")
                        for i, error in enumerate(errors[:3]):
                            print(f"    {i+1}. {error.get('check_name', 'Unknown')}: {error.get('description', '')[:50]}..." 
                                if len(error.get('description', '')) > 50 else error.get('description', ''))
                except Exception as e:
                    print(f"  Error retrieving category info: {str(e)}")
        
        print("=============================================")
        
        return st.session_state.selected_error_categories
    
    def render_specific_error_selection(self, error_repository) -> List[Dict[str, Any]]:
        """
        Render UI for selecting specific errors to include in generated code.
        
        Args:
            error_repository: Repository for accessing Java error data
            
        Returns:
            List of selected specific errors
        """
        st.subheader("Select Specific Errors")
        
        # Get all categories
        all_categories = error_repository.get_all_categories()
        build_categories = all_categories.get("build", [])
        checkstyle_categories = all_categories.get("checkstyle", [])
        
        # Selection of category type
        error_type = st.radio(
            "Error Type",
            ["Build Errors", "Code Quality Errors"],
            horizontal=True
        )

        #Filter for searching errors
        search_term = st.text_input("Search Errors", "")
        
        # Container for selected errors
        if "selected_specific_errors" not in st.session_state:
            st.session_state.selected_specific_errors = []
            
        # Display errors based on type
        if error_type == "Build Errors":
            self._display_build_errors(error_repository, build_categories,search_term)
        else:
            self._display_checkstyle_errors(error_repository, checkstyle_categories, search_term)
            
        # Show selected errors
        st.subheader("Selected Issues")
        
        if not st.session_state.selected_specific_errors:
            st.info("No specific issue selected. Random errors will be used based on categories.")
        else:
            for idx, error in enumerate(st.session_state.selected_specific_errors):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{error['type']} - {error['name']}**")
                    st.markdown(f"*{error['description']}*")
                with col2:
                    if st.button("Remove", key=f"remove_{idx}"):
                        st.session_state.selected_specific_errors.pop(idx)
                        st.rerun()
        
        return st.session_state.selected_specific_errors
        
    def _display_build_errors(self, error_repository, categories, search_term=""):
        """Display build errors with filtering"""
        # Filter errors if search term is provided
        if search_term:
            errors = [e for e in errors if search_term.lower() in e.get("error_name", "").lower() 
                    or search_term.lower() in e.get("description", "").lower()]
                 
        for category in categories:
            # Get errors for this category
            errors = error_repository.get_category_errors("build", category)
            
            if not errors:
                continue
                
            # Display category with expander
            with st.expander(f"{category} ({len(errors)} errors)"):
                for error in errors:
                    error_name = error.get("error_name", "Unknown")
                    description = error.get("description", "")
                    
                    # Check if already selected
                    is_selected = any(
                        e["type"] == "build" and e["name"] == error_name 
                        for e in st.session_state.selected_specific_errors
                    )
                    
                    # Add select button
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{error_name}**")
                        st.markdown(f"*{description}*")
                    with col2:
                        if not is_selected:
                            if st.button("Select", key=f"build_{category}_{error_name}"):
                                st.session_state.selected_specific_errors.append({
                                    "type": "build",
                                    "category": category,
                                    "name": error_name,
                                    "description": description
                                })
                                st.rerun()
                        else:
                            st.success("Selected")
                    
                    st.markdown("---")
    
    def _display_checkstyle_errors(self, error_repository, categories, search_term=""):
        """Display checkstyle errors with filtering"""
        for category in categories:
            # Get errors for this category
            errors = error_repository.get_category_errors("checkstyle", category)

            if search_term:
                errors = [e for e in errors if search_term.lower() in e.get("check_name", "").lower() 
                        or search_term.lower() in e.get("description", "").lower()]
                
            if not errors:
                continue
                
            # Display category with expander
            with st.expander(f"{category} ({len(errors)} errors)"):
                for error in errors:
                    error_name = error.get("check_name", "Unknown")
                    description = error.get("description", "")
                    
                    # Check if already selected
                    is_selected = any(
                        e["type"] == "checkstyle" and e["name"] == error_name 
                        for e in st.session_state.selected_specific_errors
                    )
                    
                    # Add select button
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{error_name}**")
                        st.markdown(f"*{description}*")
                    with col2:
                        if not is_selected:
                            if st.button("Select", key=f"checkstyle_{category}_{error_name}"):
                                st.session_state.selected_specific_errors.append({
                                    "type": "checkstyle",
                                    "category": category,
                                    "name": error_name,
                                    "description": description
                                })
                                st.rerun()
                        else:
                            st.success("Selected")
                    
                    st.markdown("---")
        
    def render_mode_selector(self) -> str:
        """
        Render the mode selector UI with improved mode switching behavior.
        
        Returns:
            Selected mode ("advanced" or "specific")
        """
        st.markdown("#### Error Selection Mode")
        
        # Create a more descriptive selection with radio buttons
        mode_options = [
            "Advanced: Select by error categories",
            "Specific: Choose exact errors to include"
        ]
        
        # Convert session state to index
        current_mode = st.session_state.error_selection_mode
        current_index = 0
        if current_mode == "specific":
            current_index = 1
        
        # Error selection mode radio buttons with CSS class for styling
        st.markdown('<div class="error-mode-radio">', unsafe_allow_html=True)
        selected_option = st.radio(
            "How would you like to select errors?",
            options=mode_options,
            index=current_index,
            key="error_mode_radio",
            label_visibility="collapsed",
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Print previous and new selection for debugging
        print(f"\n========== MODE SELECTION CHANGE ==========")
        print(f"Previous mode: {st.session_state.error_selection_mode}")
        
        # Update error selection mode based on selection
        new_mode = st.session_state.error_selection_mode
        if "Advanced" in selected_option and st.session_state.error_selection_mode != "advanced":
            new_mode = "advanced"
        elif "Specific" in selected_option and st.session_state.error_selection_mode != "specific":
            new_mode = "specific"
        
        # Only update if the mode has changed
        if new_mode != st.session_state.error_selection_mode:
            print(f"Mode changing from {st.session_state.error_selection_mode} to {new_mode}")
            
            # Store previous mode for reference
            previous_mode = st.session_state.error_selection_mode
            
            # Update the mode
            st.session_state.error_selection_mode = new_mode
                
            # Initialize or reset appropriate selections when mode changes
            if new_mode == "specific":
                # Make sure selected_specific_errors exists
                if "selected_specific_errors" not in st.session_state:
                    st.session_state.selected_specific_errors = []
        
        print(f"Current mode: {st.session_state.error_selection_mode}")
        print(f"Selected categories: {st.session_state.selected_error_categories}")
        print("=============================================")
        
        # Show help text for the selected mode
        if st.session_state.error_selection_mode == "advanced":
            st.info("Advanced mode: Select specific error categories like LogicalErrors or NamingConventionChecks. The system will randomly select errors from these categories.")
        else:
            st.info("Specific mode: Choose exactly which errors will appear in the generated code.")
        
        return st.session_state.error_selection_mode
    
    def render_code_params(self) -> Dict[str, str]:
        """
        Render code generation parameters UI with improved professional appearance.
        
        Returns:
            Dictionary with code generation parameters
        """
        # Initialize parameters if not in session state
        if "difficulty_level" not in st.session_state:
            st.session_state.difficulty_level = "Medium"
        if "code_length" not in st.session_state:
            st.session_state.code_length = "Medium"
        
        st.markdown('<div class="param-container">', unsafe_allow_html=True)
        
        # Create columns for a more compact layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="param-title">Difficulty Level</div>', unsafe_allow_html=True)
            st.markdown('<div class="param-description">Determines the complexity of errors</div>', unsafe_allow_html=True)
            difficulty_level = st.select_slider(
                "Select difficulty",
                options=["Easy", "Medium", "Hard"],
                value=st.session_state.difficulty_level,
                key="difficulty_level_select",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown('<div class="param-title">Code Length</div>', unsafe_allow_html=True)
            st.markdown('<div class="param-description" style="font-size:18px">Controls the size of the generated code</div>', unsafe_allow_html=True)
            code_length = st.select_slider(
                "Select code length",
                options=["Short", "Medium", "Long"],
                value=st.session_state.code_length,
                key="code_length_select",
                label_visibility="collapsed"
            )
        
        # Update session state
        st.session_state.difficulty_level = difficulty_level
        st.session_state.code_length = code_length
        
        # Updated difficulty explanations with more specific descriptions
        difficulty_explanation = {
            "Easy": "Very basic errors (syntax, null values, simple naming) with clear indicators, perfect for beginners",
            "Medium": "Mix of basic and intermediate errors requiring code reading skills, suitable for those with some Java experience",
            "Hard": "Subtle logical errors, edge cases, and complex bugs that require deeper Java knowledge and debugging skills"
        }

        # Updated length explanation with beginner-friendly sizes
        length_explanation = {
            "Short": "15-30 lines of code, a single simple class with 1-2 methods - ideal for beginners",
            "Medium": "40-80 lines, a class with 3-5 methods - good for intermediate practice",
            "Long": "100-150 lines, 1-2 classes with several methods - for advanced review practice"
        }
        
        st.markdown('<div class="param-value">', unsafe_allow_html=True)
        st.markdown(f"**Difficulty:** {difficulty_level} - {difficulty_explanation[difficulty_level]}", unsafe_allow_html=True)
        st.markdown(f"**Length:** {code_length} - {length_explanation[code_length]}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            "difficulty_level": difficulty_level.lower(),
            "code_length": code_length.lower()
        }