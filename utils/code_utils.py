"""
Utility functions for code generation and processing in the Java Code Review System.

This module provides shared functionality for generating prompts, 
extracting code from responses, and handling error comments.
"""

import re
import random
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Optimized prompting strategies for the Java Peer Review Training System.

This module provides streamlined prompts for code generation, evaluation,
and review analysis to reduce token usage while maintaining quality.
"""

def add_line_numbers(code: str) -> str:
    """
    Add line numbers to code snippet.
    
    Args:
        code: The code snippet to add line numbers to
        
    Returns:
        Code with line numbers
    """
    lines = code.splitlines()
    max_line_num = len(lines)
    padding = len(str(max_line_num))
    
    # Create a list of lines with line numbers
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        # Format line number with consistent padding
        line_num = str(i).rjust(padding)
        numbered_lines.append(f"{line_num} | {line}")
    
    return "\n".join(numbered_lines)

def create_code_generation_prompt(code_length: str, difficulty_level: str, selected_errors: list, domain: str = None, include_error_annotations: bool = True) -> str:
    """
    Create a concise prompt for generating Java code with intentional errors.
    Enhanced to emphasize the exact number of errors required and ensure one per error type.
    
    Args:
        code_length: Length of code (short, medium, long)
        difficulty_level: Difficulty level (easy, medium, hard)
        selected_errors: List of errors to include in the code
        domain: Domain context for the code
        include_error_annotations: Whether to include error annotations
        
    Returns:
        Optimized prompt string for LLM
    """
    # Define code complexity by length
    complexity = {
        "short": "1 simple class with 1-2 basic methods (15-30 lines total)",
        "medium": "1 class with 3-5 methods of moderate complexity (40-80 lines total)",
        "long": "1-2 classes with 4-8 methods and clear relationships (100-150 lines total)"
    }.get(str(code_length).lower(), "1 class with methods")
    
    # Count the number of errors
    error_count = len(selected_errors)
    
    # Format errors concisely with only essential information
    error_list = []
    for i, error in enumerate(selected_errors, 1):
        error_type = error.get("type", "unknown").upper()
        name = error.get("name", "unknown")
        description = error.get("description", "")
        implementation_guide = error.get("implementation_guide", "")
        
        error_entry = f"{i}. {error_type} - {name}: {description}"
        if implementation_guide:
            error_entry += f"\nImplementation: {implementation_guide}"
        
        error_list.append(error_entry)
    
    # Join errors with clear separation
    error_instructions = "\n\n".join(error_list)
    
    # Add difficulty-specific instructions
    difficulty_instructions = ""
    if difficulty_level.lower() == "easy":
        difficulty_instructions = """
            BEGINNER-FRIENDLY CODE REQUIREMENTS:
            - Use very descriptive variable/method names (studentName, calculateTotal)
            - Keep methods short (3-10 lines each) and focused on a single task
            - Use basic control structures (if/else, simple loops) with clear conditions
            - Include helpful comments explaining the code's purpose
            - Avoid complex nested structures or advanced Java features
            - Make errors relatively obvious for educational purposes
            - Implement errors in a way that beginners can reasonably identify them
            """
    elif difficulty_level.lower() == "medium":
        difficulty_instructions = """
            INTERMEDIATE-LEVEL CODE REQUIREMENTS:
            - Use a mix of simple and moderately complex code structures
            - Include a variety of control structures and data types
            - Keep methods reasonably sized (5-15 lines)
            - Implement some errors that require careful reading to identify
            - Add appropriate documentation where needed
            - Create realistic code that might appear in a small application
            - Balance obvious errors with some more subtle ones
            """
    else:  # hard
        difficulty_instructions = """
            ADVANCED-LEVEL CODE REQUIREMENTS:
            - Create more sophisticated code structures with appropriate complexity
            - Implement errors that might be hidden in logical flow or edge cases
            - Use a variety of Java features and design patterns when appropriate
            - Challenge the student to think deeply about the code
            - Include subtle errors that require careful analysis to identify
            - Create realistic code that follows good structure despite the errors
            - Implement errors that interact with each other in non-obvious ways
            """
    
    domain_str = domain or "general"
    
    # Create a focused prompt with clear role definition and verification steps
    prompt = f"""You are an expert Java programming instructor creating educational code with specific deliberate errors for students to practice code review skills.

        MAIN TASK:
        Generate a {code_length} Java program for a {domain_str} system that contains EXACTLY {error_count} intentional errors for a code review exercise.

        CODE STRUCTURE REQUIREMENTS:
        - Create {complexity}
        - Make the code realistic, well-structured, and appropriate for a {domain_str} application
        - Follow standard Java conventions for all correct parts of the code
        - The code should look professional except for the deliberate errors

        {difficulty_instructions}

        ERROR IMPLEMENTATION REQUIREMENTS:
        - Implement EXACTLY {error_count} errors - this is CRITICAL (no more, no fewer)
        - Only implement the SPECIFIC errors listed below
        - Each error must be an actual Java error, not just a comment
        - In the annotated version, mark each error with a comment: // ERROR: [TYPE] - [NAME] - [Brief explanation]
        - NEVER add comments like "// added to fix" or "// this is incorrect" - the errors are meant to remain as errors!
        - Ensure errors are findable through code review (not just runtime errors)

        EXACTLY {error_count} ERRORS TO IMPLEMENT:

        {error_instructions}

        VERIFICATION CHECKLIST (COMPLETE BEFORE SUBMITTING):
        - [ ] Code follows the {code_length}/{difficulty_level} complexity requirements
        - [ ] Code is realistic and appropriate for a {domain_str} application
        - [ ] EXACTLY {error_count} errors are implemented (no more, no fewer)
        - [ ] Each implemented error matches one from the requested list
        - [ ] All errors are marked with appropriate comments in the annotated version
        - [ ] The clean version has the same errors but without the comments
        - [ ] Both versions would compile (except for deliberate compilation errors)

        OUTPUT FORMAT:
        1. First, provide the ANNOTATED VERSION with error comments:
        ```java-annotated
        // Your code with error annotations
        ```

        2. Then, provide the CLEAN VERSION without any error comments:
        ```java-clean
        // The same code with the same errors but no error annotations
        ```

        IMPORTANT: Verify you have implemented EXACTLY {error_count} errors before completing.
        """
    
    return prompt

def create_evaluation_prompt(code: str, requested_errors: list) -> str:
    """
    Create a clear and concise prompt for evaluating whether code contains required errors.
    Improved with detailed evaluation criteria and structured output format.
    """
    # Count the exact number of requested errors
    error_count = len(requested_errors)
    
    # Format requested errors clearly
    error_list = []
    for i, error in enumerate(requested_errors, 1):
        error_type = error.get("type", "").upper()
        name = error.get("name", "")
        description = error.get("description", "")
        error_list.append(f"{i}. {error_type} - {name}: {description}")
    
    error_instructions = "\n".join(error_list)
    
    # Create focused evaluation prompt with clear role definition
    prompt = f"""As a Java code quality expert, your task is to analyze Java code to determine if it correctly implements specific requested errors.

            MAIN TASK:
            Evaluate if the provided Java code correctly implements EXACTLY {error_count} specific errors that were requested.

            CODE TO EVALUATE:
            ```java
            {code}
            ```

            THE {error_count} SPECIFIC ERRORS THAT SHOULD BE PRESENT:
            {error_instructions}

            EVALUATION INSTRUCTIONS:
            1. Examine the code line by line, identifying each error that matches the requested list
            2. For each error you find, note:
            - The specific error type and name
            - The exact line number(s) where it appears
            - A brief code segment showing the error
            - A concise explanation of why it matches the requested error
            3. Check if any requested errors are missing from the code
            4. For valid implementation, the code must contain EXACTLY {error_count} errors - no more, no fewer

            RESPONSE FORMAT:
            Your evaluation must be returned in this JSON format:

            ```json
            {{
            "found_errors": [
                {{
                "error_type": "BUILD",  
                "error_name": "NullPointerException",
                "line_number": 42,
                "code_segment": "String str = null; int length = str.length();",
                "explanation": "This code will cause a NullPointerException because it calls length() on a null String"
                }}
                // List all implemented errors that match the requested list
            ],
            "missing_errors": [
                {{
                "error_type": "CHECKSTYLE",
                "error_name": "MemberName",
                "explanation": "The code doesn't contain any variable names that violate member naming conventions"
                }}
                // List all requested errors that aren't implemented
            ],
            "valid": true,  // Set to true ONLY if ALL requested errors are implemented, no more and no fewer
            "feedback": "The code successfully implements all {error_count} requested errors."  // Provide brief overall assessment
            }}
            ```

            VERIFICATION CHECKLIST:
            - Confirm that each found error truly matches the corresponding requested error
            - Verify that the total count of found errors is EXACTLY {error_count} for validity
            - Double-check any errors you believe are missing to ensure they're truly absent
            - Ensure your JSON response is properly formatted for processing

            IMPORTANT: Focus solely on the specified error types and names, not general code quality issues.
            """
    
    return prompt

def create_regeneration_prompt(code: str, domain: str, missing_errors: list, found_errors: list, requested_errors: list) -> str:
    """
    Create a focused prompt for regenerating code with missing errors and removing extra errors.
    Enhanced to provide clear instructions for exact error requirements.
    """
    # Total requested errors count
    total_requested = len(requested_errors)
    
    # Create detailed instructions for missing errors
    missing_instructions = []
    for error_key in missing_errors:
        # Find the full error details
        for error in requested_errors:
            error_type = error.get("type", "").upper()
            name = error.get("name", "")
            key_match = f"{error_type} - {name}" == error_key
            
            # Also check for partial matches if exact match fails
            if not key_match and error_key:
                # Try to match on error name alone
                if name and name.lower() in error_key.lower():
                    key_match = True
                # Try to match on error type alone
                elif error_type and error_type.lower() in error_key.lower():
                    key_match = True
            
            if key_match:
                guide = error.get("implementation_guide", "")
                description = error.get("description", "")
                
                instruction = f"{error_type} - {name}"
                if description:
                    instruction += f": {description}"
                if guide:
                    instruction += f"\nImplementation: {guide}"
                missing_instructions.append(instruction)
                break
    
    # Format missing and found errors
    missing_text = "\n".join(f"- {instr}" for instr in missing_instructions)
    found_text = "\n".join(f"- {err}" for err in found_errors)
    
    # Create improved prompt with clearer instructions and error verification steps
    prompt = f"""You are an educational Java error creator who intentionally introduces specific errors in code for teaching purposes.

        TASK:
        Modify this Java code to have EXACTLY {total_requested} errors - no more, no fewer.
        The code must contain ONLY the specific errors requested below.

        ORIGINAL CODE DOMAIN: {domain}

        MISSING ERRORS - INTENTIONALLY add these errors (do NOT fix or solve them):
        {missing_text if missing_text else "No missing errors - all requested errors are already implemented."}

        EXISTING ERRORS TO KEEP - Do not modify these errors:
        {found_text if found_text else "No correctly implemented errors found."}

        VERY IMPORTANT INSTRUCTIONS:
        1. Focus on implementing EXACTLY the requested errors
        2. NEVER add comments like "// added to fix", "// fixed", or "// corrected" - these errors are meant to remain as errors!
        3. Do not change the domain or structure of the code
        4. Errors must be actual Java errors, not just comments about errors
        5. Use EXACTLY the same {domain} domain and maintain the original code structure
        6. For each error you add, include a comment in the format: // ERROR: [TYPE] - [NAME] - [Brief explanation]
        7. Do NOT try to improve or fix the code - it should contain intentional bugs for educational purposes
        8. The whole purpose is to create flawed code that students will learn to identify problems in

        VERIFICATION STEPS (DO THIS BEFORE SUBMITTING):
        1. Count the total number of errors in your code, confirm it's EXACTLY {total_requested}
        2. Verify each missing error from the list is now implemented
        3. Confirm all existing errors that should be kept are still present and unchanged
        4. Ensure any extra errors have been removed

        PROVIDE TWO VERSIONS OF THE CODE:
        1. First, provide the ANNOTATED VERSION with error comments, marked with:
        ```java-annotated
        // Your code with intentional errors and error annotations
        ```

        2. Then, provide the CLEAN VERSION without any error comments, marked with:
        ```java-clean
        // The same code with the same intentional errors but no error comments
        ```

        ORIGINAL CODE:
        ```java
        {code}
        ```
        """
    
    return prompt

def create_review_analysis_prompt(code: str, known_problems: list, student_review: str) -> str:
    """
    Create an optimized prompt for analyzing student code reviews.
    Enhanced with educational assessment focus and better structured output requirements.
    """
    # Count known problems
    problem_count = len(known_problems)
    
    # Format known problems clearly
    problems_text = "\n".join(f"- {problem}" for problem in known_problems)
    
    # Create focused analysis prompt with educational assessment role
    prompt = f"""You are an educational assessment specialist analyzing a student's Java code review skills.

                MAIN TASK:
                Analyze the student's code review against a set of known issues to evaluate their code review effectiveness.

                CODE BEING REVIEWED:
                ```java
                {code}
                ```

                {problem_count} KNOWN ISSUES IN THE CODE:
                {problems_text}

                STUDENT'S REVIEW SUBMISSION:
                ```
                {student_review}
                ```

                ANALYSIS INSTRUCTIONS:
                1. Carefully read both the code and the student's review
                2. Identify which of the known issues the student correctly found
                3. Note which known issues the student missed
                4. Identify any false positives (things the student flagged as issues that aren't actual problems)
                5. Evaluate the review quality (accuracy, completeness, clarity, and specificity)
                6. Determine if the review is sufficient (>= 60% of issues correctly identified)

                RESPONSE REQUIREMENTS:
                Provide your analysis in JSON format with these components:

                ```json
                {{
                "identified_problems": [
                    {{
                    "problem": "SPECIFIC KNOWN ISSUE TEXT",
                    "student_comment": "STUDENT'S RELEVANT COMMENT",
                    "accuracy": 0.9,
                    "feedback": "Brief feedback on this identification"
                    }}
                    // Include all correctly identified issues
                ],
                "missed_problems": [
                    {{
                    "problem": "SPECIFIC KNOWN ISSUE TEXT",
                    "hint": "A helpful educational hint for finding this type of issue"
                    }}
                    // Include all missed issues
                ],
                "false_positives": [
                    {{
                    "student_comment": "STUDENT'S INCORRECT COMMENT",
                    "explanation": "Why this isn't actually an issue"
                    }}
                    // Include any incorrect identifications
                ],
                "identified_count": 3,  // Number of correctly identified issues
                "total_problems": {problem_count},  // Total number of known issues
                "identified_percentage": 60.0,  // Percentage of issues correctly identified
                "review_quality_score": 7.5,  // Score from 1-10 rating review quality
                "review_sufficient": true,  // true if >= 60% of issues identified
                "feedback": "Overall assessment with specific improvement suggestions"
                }}
                ```

                EVALUATION CRITERIA:
                - For matching student comments to known issues, look for:
                - Correct identification of the issue type
                - Accurate location (line number or description)
                - Understanding of why it's a problem
                - Consider partial credit if they identified an issue but misunderstood it
                - A review is sufficient if the student correctly identified at least 60% of known issues

                TIPS FOR ANALYSIS:
                - Be thorough in examining every part of the student's review
                - Be generous in matching student comments to issues if they show understanding
                - Provide educational feedback that helps the student improve their code review skills
                - If the student uses different terminology but correctly identifies an issue, count it as correct
                """
    
    return prompt

def create_feedback_prompt(code: str, known_problems: list, review_analysis: dict) -> str:
    """
    Create an optimized prompt for generating concise, focused guidance on student reviews.
    Enhanced with clearer educational goals and example output.
    """
    # Extract data from review analysis
    identified = review_analysis.get("identified_count", 0)
    total = review_analysis.get("total_problems", len(known_problems))
    accuracy = review_analysis.get("identified_percentage", 0)
    iteration = review_analysis.get("iteration_count", 1)
    max_iterations = review_analysis.get("max_iterations", 3)
    remaining = review_analysis.get("remaining_attempts", max_iterations - iteration)
    
    # Format identified problems
    identified_problems = review_analysis.get("identified_problems", [])
    identified_text = ""
    for problem in identified_problems:
        if isinstance(problem, dict):
            problem_text = problem.get("problem", "")
            identified_text += f"- {problem_text}\n"
        else:
            identified_text += f"- {problem}\n"
    
    # Format missed problems
    missed_problems = review_analysis.get("missed_problems", [])
    missed_text = ""
    for problem in missed_problems:
        if isinstance(problem, dict):
            problem_text = problem.get("problem", "")
            missed_text += f"- {problem_text}\n"
        else:
            missed_text += f"- {problem}\n"
    
    # Create focused feedback prompt with educational coach role
    prompt = f"""As a Java mentor providing targeted code review guidance, create concise feedback for a student.

                CONTEXT:
                - Student completed review attempt {iteration} of {max_iterations}
                - Found {identified}/{total} issues ({accuracy:.1f}%)
                - {remaining} review attempts remaining

                CORRECTLY IDENTIFIED ISSUES:
                {identified_text or "None"}

                MISSED ISSUES:
                {missed_text or "None - great job!"}

                TASK:
                Create brief, specific guidance (3-4 sentences max) to help the student find more issues in their next review attempt.

                GUIDANCE REQUIREMENTS:
                1. Be extremely concise and focused (max 3-4 short sentences)
                2. Target the most important 1-2 areas for improvement
                3. Provide specific, actionable strategies (what to look for)
                4. Be encouraging but direct
                5. Focus only on helping them find missed issues, not general code review skills

                EXAMPLE GOOD GUIDANCE:
                "Look more carefully at method parameters and return types. Several issues involve type mismatches that can be spotted by comparing declared types with actual values. Also check for proper null handling before method calls."

                EXAMPLE POOR GUIDANCE (too general):
                "Keep trying to find more issues. There are several problems in the code that you missed. Try to be more thorough in your next review attempt."

                RESPONSE FORMAT:
                Provide ONLY the guidance text with no introduction or explanation.
                """
    
    return prompt

def extract_both_code_versions(response) -> Tuple[str, str]:
    """
    Extract both annotated and clean code versions from LLM response.
    Enhanced to better handle Groq response format differences.
    
    Args:
        response: Text response from LLM or AIMessage/ChatMessage object
        
    Returns:
        Tuple of (annotated_code, clean_code)
    """
    # Check for None or empty response
    if not response:
        return "", ""
    
    # Handle AIMessage or similar objects (from LangChain)
    if hasattr(response, 'content'):
        # Extract the content from the message object
        response_text = response.content
    elif isinstance(response, dict) and 'content' in response:
        # Handle dictionary-like response
        response_text = response['content']
    else:
        # Assume it's already a string
        response_text = str(response)
    
    # Handle Groq-specific response format
    # Groq often wraps content differently, so check for that pattern
    if "content=" in response_text and not response_text.startswith("```"):
        # Extract just the content part
        response_text = response_text.replace("content=", "")
        # Remove any leading/trailing quotes if present
        if (response_text.startswith('"') and response_text.endswith('"')) or \
           (response_text.startswith("'") and response_text.endswith("'")):
            response_text = response_text[1:-1]
    
    # Extract annotated version with java-annotated tag
    annotated_pattern = r'```java-annotated\s*(.*?)\s*```'
    annotated_matches = re.findall(annotated_pattern, response_text, re.DOTALL)
    annotated_code = annotated_matches[0] if annotated_matches else ""
    
    # Extract clean version with java-clean tag
    clean_pattern = r'```java-clean\s*(.*?)\s*```'
    clean_matches = re.findall(clean_pattern, response_text, re.DOTALL)
    clean_code = clean_matches[0] if clean_matches else ""
    
    # Fallbacks if specific tags aren't found
    if not annotated_code:
        # Try to find any java code block for annotated version
        java_pattern = r'```java\s*(.*?)\s*```'
        java_matches = re.findall(java_pattern, response_text, re.DOTALL)
        if java_matches:
            annotated_code = java_matches[0]
        else:
            # Last resort: look for any code block
            any_code_pattern = r'```\s*(.*?)\s*```'
            any_matches = re.findall(any_code_pattern, response_text, re.DOTALL)
            if any_matches:
                # Use the largest code block
                annotated_code = max(any_matches, key=len)
    
    # For Groq responses: If we found annotated but no clean code, create clean code by removing error comments
    if annotated_code and not clean_code:
        # Remove lines with error comments
        clean_lines = []
        for line in annotated_code.splitlines():
            if "// ERROR:" not in line:
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)
    
    # Log detailed information if extraction failed
    if not annotated_code:
        logger.warning(f"Failed to extract annotated code from response text: {response_text[:200]}...")
    if not clean_code:
        logger.warning(f"Failed to extract clean code from response text: {response_text[:200]}...")
    
    return annotated_code, clean_code

def generate_comparison_report(evaluation_errors: List[str], review_analysis: Dict[str, Any], 
                              review_history: List[Dict[str, Any]] = None, llm = None) -> str:
    """
    Generate a comparison report showing progress across review attempts.
    Uses an LLM when available, with fallback to static generation.
    
    Args:
        evaluation_errors: List of errors found by the evaluation
        review_analysis: Analysis of the latest student review
        review_history: History of all review attempts
        llm: Optional language model to generate the report
        
    Returns:
        Formatted comparison report
    """
    # If LLM is provided, use it to generate the report
    if llm:
        try:
            # Create the prompt for the LLM
            prompt = create_comparison_report_prompt(evaluation_errors, review_analysis, review_history)
            
            # Generate the report with the LLM
            response = llm.invoke(prompt)
            
            # Process the response
            if hasattr(response, 'content'):
                report = response.content
            elif isinstance(response, dict) and 'content' in response:
                report = response['content']
            else:
                report = str(response)
            
            # Clean up the report
            report = report.replace('\\n', '\n')
            
            return report
        except Exception as e:
            # Log the error
            logger.error(f"Error generating comparison report with LLM: {str(e)}")
            # Fall back to static generation
            return generate_comparison_report_fallback(evaluation_errors, review_analysis, review_history)
    else:
        # If no LLM is provided, use static generation
        return generate_comparison_report_fallback(evaluation_errors, review_analysis, review_history)

def create_comparison_report_prompt(evaluation_errors: List[str], review_analysis: Dict[str, Any], review_history: List[Dict[str, Any]] = None) -> str:
    """
    Create a prompt for generating a comparison report with an LLM.
    """
    # Extract performance metrics from latest review
    identified_problems = review_analysis.get("identified_problems", [])
    missed_problems = review_analysis.get("missed_problems", [])
    false_positives = review_analysis.get("false_positives", [])
    
    # Get total problems count
    total_problems = (review_analysis.get("total_problems", 0) or 
                       review_analysis.get("original_error_count", 0) or 
                       len(evaluation_errors))
    
    # Calculate metrics
    identified_count = len(identified_problems)
    accuracy = (identified_count / total_problems * 100) if total_problems > 0 else 0
    
    # Format the problems for the prompt
    identified_str = []
    for problem in identified_problems:
        if isinstance(problem, dict) and "problem" in problem:
            identified_str.append(problem["problem"])
        elif isinstance(problem, str):
            identified_str.append(problem)
    
    missed_str = []
    for problem in missed_problems:
        if isinstance(problem, dict) and "problem" in problem:
            missed_str.append(problem["problem"])
        elif isinstance(problem, str):
            missed_str.append(problem)
    
    false_str = []
    for problem in false_positives:
        if isinstance(problem, dict) and "student_comment" in problem:
            false_str.append(problem["student_comment"])
        elif isinstance(problem, str):
            false_str.append(problem)
    
    # Format identified problems for the prompt
    identified_text = "\n".join(f"- {p}" for p in identified_str)
    missed_text = "\n".join(f"- {p}" for p in missed_str)
    false_positive_text = "\n".join(f"- {p}" for p in false_str)
    
    # Create progress tracking info if multiple attempts exist
    progress_info = ""
    if review_history and len(review_history) > 1:
        progress_info = "## Progress Across Attempts\n\n"
        
        for i, review in enumerate(review_history, 1):
            analysis = review.get("review_analysis", {})
            found = analysis.get("identified_count", 0)
            acc = analysis.get("identified_percentage", 0)
            progress_info += f"Attempt {i}: Found {found}/{total_problems} issues ({acc:.1f}%)\n"
        
        # Compare first vs. latest attempt
        first = review_history[0].get("review_analysis", {})
        first_found = first.get("identified_count", 0)
        first_acc = first.get("identified_percentage", 0)
        
        if accuracy > first_acc:
            improvement = accuracy - first_acc
            progress_info += f"\nImprovement: +{improvement:.1f}% from first attempt\n"
    
    # Create the prompt for the LLM
    prompt = f"""You are an educational assessment expert creating a detailed, informative code review feedback report for a Java programming student.

                CONTEXT:
                The student has conducted a code review exercise, identifying errors in a Java code snippet. Your task is to create a comprehensive, educational report on their performance.

                PERFORMANCE METRICS:
                - Total issues in the code: {total_problems}
                - Issues correctly identified: {identified_count} ({accuracy:.1f}%)
                - Issues missed: {len(missed_str)}
                - False positives (things incorrectly flagged as issues): {len(false_str)}

                CORRECTLY IDENTIFIED ISSUES:
                {identified_text or "None - the student didn't identify any correct issues."}

                MISSED ISSUES:
                {missed_text or "None - the student identified all issues correctly!"}

                FALSE POSITIVES:
                {false_positive_text or "None - the student didn't identify any false issues."}

                {progress_info}

                REPORT REQUIREMENTS:
                1. Create a comprehensive educational report in markdown format
                2. Include these sections:
                - Performance Summary (with metrics and overall assessment)
                - Correctly Identified Issues (with praise for what they found correctly)
                - Missed Issues (with educational explanations of why they matter)
                - False Positives (if any, with explanations of why these aren't actual issues)
                - Progress Analysis (if multiple attempts, analyzing their improvement)
                - Tips for Improvement (specific, actionable advice based on their performance)

                3. Be educational and constructive, not just evaluative
                4. Use a warm, encouraging tone while maintaining honesty about areas for improvement
                5. Focus on helping them become a better code reviewer, not just scoring this attempt
                6. Highlight patterns in what they missed or found to help them improve systematically
                7. Include specific Java code review tips relevant to their performance
                8. Make the report visually readable with appropriate markdown formatting

                IMPORTANT FORMATTING:
                - Use markdown for clear organization (headers, bullet points, etc.)
                - Format code snippets in markdown code blocks if referring to specific code
                - Use bold or italic text for emphasis where appropriate
                - Keep paragraphs reasonably short for readability
                """
    
    return prompt

def generate_comparison_report_fallback(evaluation_errors: List[str], review_analysis: Dict[str, Any], 
                              review_history: List[Dict[str, Any]] = None) -> str:
    """
    Generate a static comparison report showing progress across review attempts.
    Used as a fallback when LLM generation is not available.
    
    Args:
        evaluation_errors: List of errors found by the evaluation
        review_analysis: Analysis of the latest student review
        review_history: History of all review attempts
        
    Returns:
        Formatted comparison report
    """
    # Extract performance metrics from latest review
    identified_problems = review_analysis.get("identified_problems", [])
    missed_problems = review_analysis.get("missed_problems", [])
    false_positives = review_analysis.get("false_positives", [])
    
    # Get total problems count
    total_problems = (review_analysis.get("total_problems", 0) or 
                     review_analysis.get("original_error_count", 0) or 
                     len(evaluation_errors))
    
    # Calculate metrics
    identified_count = len(identified_problems)
    accuracy = (identified_count / total_problems * 100) if total_problems > 0 else 0
    
    # Convert all problems to strings
    identified_str = [str(p) if not isinstance(p, str) else p for p in identified_problems]
    missed_str = [str(p) if not isinstance(p, str) else p for p in missed_problems]
    false_str = [str(p) if not isinstance(p, str) else p for p in false_positives]
    
    # Build report with markdown
    report = "# Code Review Assessment\n\n"
    
    # Add progress tracking if multiple attempts exist
    if review_history and len(review_history) > 1:
        report += "## Progress Across Attempts\n\n"
        report += "| Attempt | Issues Found | Accuracy |\n"
        report += "|---------|--------------|----------|\n"
        
        for i, review in enumerate(review_history, 1):
            analysis = review.get("review_analysis", {})
            found = analysis.get("identified_count", 0)
            acc = analysis.get("identified_percentage", 0)
            report += f"| {i} | {found}/{total_problems} | {acc:.1f}% |\n"
        
        # Compare first vs. latest attempt
        first = review_history[0].get("review_analysis", {})
        first_found = first.get("identified_count", 0)
        first_acc = first.get("identified_percentage", 0)
        
        if accuracy > first_acc:
            improvement = accuracy - first_acc
            report += f"\n📈 **Improvement**: +{improvement:.1f}% from first attempt\n\n"
    
    # Performance summary
    report += f"## Final Review Performance\n\n"
    report += f"**Score:** {identified_count}/{total_problems} issues identified ({accuracy:.1f}%)\n\n"
    
    # Issues identified in latest attempt
    if identified_str:
        report += "## Issues Correctly Identified\n\n"
        for i, problem in enumerate(identified_str, 1):
            report += f"✅ **{i}.** {problem}\n\n"
    
    # Issues missed in latest attempt
    if missed_str:
        report += "## Issues Missed\n\n"
        for i, problem in enumerate(missed_str, 1):
            report += f"❌ **{i}.** {problem}\n\n"
            
            # Add specific guidance for missed issues
            problem_lower = problem.lower()
            if "null" in problem_lower:
                report += "*Tip: Check for null pointer handling before method calls*\n\n"
            elif "name" in problem_lower or "convention" in problem_lower:
                report += "*Tip: Verify variable/class naming conventions (camelCase, PascalCase)*\n\n"
            elif "equals" in problem_lower or "==" in problem_lower:
                report += "*Tip: Look for object equality issues (.equals() vs ==)*\n\n"
    
    # False positives
    if false_str:
        report += "## False Positives\n\n"
        for i, problem in enumerate(false_str, 1):
            report += f"⚠️ **{i}.** {problem}\n\n"
    
    # New knowledge gained (if multiple attempts)
    if review_history and len(review_history) > 1:
        # Get identified issues from first attempt
        first_review = review_history[0].get("review_analysis", {})
        first_identified = first_review.get("identified_problems", [])
        first_identified_str = [str(p) if not isinstance(p, str) else p for p in first_identified]
        
        # Find newly identified issues in the latest attempt
        new_findings = [p for p in identified_str if p not in first_identified_str]
        
        if new_findings:
            report += "## New Issues Found\n\n"
            report += "*Issues you identified in your latest attempt that you missed initially:*\n\n"
            for i, problem in enumerate(new_findings, 1):
                report += f"🔍 **{i}.** {problem}\n\n"
    
    # Quick tip
    report += "\n**Tip for next time:** Use format `Line X: [Error Type] - Description` in your reviews.\n"
    
    return report

def process_llm_response(response):
    """
    Process LLM response to handle different formats from different providers
    with improved error handling and type safety.
    
    Args:
        response: Response from LLM (string, AIMessage, or dict)
        
    Returns:
        Cleaned string content
    """
    # Handle None case
    if response is None:
        return ""
    
    try:
        # Extract content based on response type
        if hasattr(response, 'content'):
            # AIMessage or similar object from LangChain
            content = response.content
        elif isinstance(response, dict) and 'content' in response:
            # Dictionary with content key
            content = response['content']
        else:
            # Assume it's already a string
            content = str(response)
        
        # Fix common formatting issues:
        
        # 1. Remove any 'content=' prefix if present (common in Groq debug output)
        if content.startswith('content='):
            content = content.replace('content=', '', 1)
        
        # 2. Fix escaped newlines and quotes
        content = content.replace('\\n', '\n')
        content = content.replace('\\"', '"')
        content = content.replace('\\\'', '\'')
        
        # 3. Remove any surrounding quotes that might have been added
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
        
        # 4. Fix markdown formatting issues
        content = re.sub(r'\*\*(.+?)\*\*', r'**\1**', content)  # Fix bold formatting
        
        # 5. Clean up any raw escape sequences for newlines
        content = re.sub(r'(?<!\\)\\n', '\n', content)
        content = re.sub(r'\\\\n', '\\n', content)  # Preserve intentional \n in code
        
        # 6. Fix any metadata that might have leaked into the content
        content = re.sub(r'response_metadata=\{.*\}', '', content)
        content = re.sub(r'additional_kwargs=\{.*\}', '', content)
        
        return content
    except Exception as e:
        logger.error(f"Error processing LLM response: {str(e)}")
        # Return a safe default
        if response is not None:
            try:
                return str(response)
            except:
                pass
        return ""

def get_error_count_from_state(state: Any, difficulty_level: str = "medium") -> int:
    """
    Get error count from the state object or parameters.
    Replaces the fixed get_error_count_for_difficulty function.
    
    Args:
        state: State object that might contain error count info
        difficulty_level: Fallback difficulty level if state doesn't have count
        
    Returns:
        Number of errors to use
    """
    # First try to get error count from selected_specific_errors if available
    if hasattr(state, 'selected_specific_errors') and state.selected_specific_errors:
        return len(state.selected_specific_errors)
    
    # Next try to get from original_error_count if it's been set
    if hasattr(state, 'original_error_count') and state.original_error_count > 0:
        return state.original_error_count
    
    # If we have selected error categories, use their count
    if hasattr(state, 'selected_error_categories'):
        selected_categories = state.selected_error_categories
        if selected_categories:
            build_errors = selected_categories.get("build", [])
            checkstyle_errors = selected_categories.get("checkstyle", [])
            # Use at least one error per selected category
            category_count = len(build_errors) + len(checkstyle_errors)
            if category_count > 0:
                return max(category_count, 2)  # Ensure at least 2 errors
    
    # Finally fall back to difficulty-based default if all else fails
    difficulty_map = {
        "easy": 2,
        "medium": 4,
        "hard": 6
    }
    return difficulty_map.get(str(difficulty_level).lower(), 4)