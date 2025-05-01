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

def create_regeneration_prompt(code: str, domain: str, missing_errors: list, found_errors: list, requested_errors: list, extra_errors: list = None) -> str:
    """
    Create a focused prompt for regenerating code with missing errors and removing extra errors.
    Enhanced to provide clear instructions for exact error requirements.
    
    Args:
        code: The original code to improve
        domain: Domain of the code (must be consistent with original)
        missing_errors: List of error keys that need to be implemented
        found_errors: List of error keys already implemented correctly
        requested_errors: Full list of requested error dictionaries
        extra_errors: List of extra errors that should be removed
        
    Returns:
        Optimized regeneration prompt
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
    
    # Create text for extra errors that need to be removed
    extra_text = ""
    if extra_errors and len(extra_errors) > 0:
        extra_list = "\n".join(f"- {err}" for err in extra_errors)
        extra_text = f"""
        EXTRA ERRORS TO REMOVE:
        These errors were not requested and should be removed from the code:
        {extra_list}
        """
    
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

            {extra_text}

            VERY IMPORTANT INSTRUCTIONS:
            1. The final code MUST contain EXACTLY {total_requested} errors - no more, no less
            2. NEVER add comments like "// added to fix", "// fixed", or "// corrected" - these errors are meant to remain as errors!
            3. Do not add any errors which are not in the requested list
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

def create_evaluation_prompt(code: str, requested_errors: list) -> str:
    """
    Create a clear and concise prompt for evaluating whether code contains required errors.
    
    Args:
        code: The generated Java code
        requested_errors: List of errors that should be implemented
        
    Returns:
        Optimized evaluation prompt
    """
    # Count the exact number of requested errors
    error_count = len(requested_errors)
    
    # Format requested errors clearly
    error_list = []
    for error in requested_errors:
        error_type = error.get("type", "").upper()
        name = error.get("name", "")
        description = error.get("description", "")
        error_list.append(f"{error_type} - {name}: {description}")
    
    error_instructions = "\n".join(f"{i+1}. {error}" for i, error in enumerate(error_list))
    
    # Create focused evaluation prompt with clear role definition
    prompt = f"""You are a Java code assessment expert. Your task is to evaluate a Java code sample and determine if it correctly implements EXACTLY {error_count} specific errors that were requested.

        JAVA CODE TO EVALUATE:
        ```java
        {code}
        ```

        EXACTLY {error_count} REQUESTED ERRORS:
        {error_instructions}

        YOUR EVALUATION TASK:
        1. Analyze the code to find which of the requested errors are correctly implemented
        2. Identify the exact line number and code segment for each implemented error
        3. Determine if any requested errors are missing from the code
        4. Check if there are any extra errors beyond the {error_count} that were requested
        5. Return a JSON response with your findings

        YOUR RESPONSE MUST BE IN THIS JSON FORMAT:
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
            // Include all implemented errors that match the requested list
        ],
        "missing_errors": [
            {{
            "error_type": "CHECKSTYLE",
            "error_name": "MemberName",
            "explanation": "The code doesn't contain any variable names that violate member naming conventions"
            }}
            // Include all requested errors that are not implemented
        ],
        "extra_errors": [
            {{
            "error_type": "BUILD",
            "error_name": "Unreported Exception",
            "line_number": 25,
            "explanation": "This error was not in the requested list"
            }}
            // Include any errors found in the code that weren't requested
        ],
        "valid": false,
        "feedback": "The code contains 3 of 4 requested errors, is missing 1 requested error, and has 2 extra errors not requested."
        }}
        ```

        IMPORTANT CRITERIA:
        - The code must contain EXACTLY {error_count} errors - no more, no fewer
        - Set "valid" to true ONLY if ALL requested errors are implemented AND there are NO extra errors
        - Provide specific line numbers and code segments for all found errors
        - In the "feedback" field, clearly state how many errors were found, how many are missing, and how many extra errors exist
        """
    
    return prompt

def create_code_generation_prompt(code_length: str, difficulty_level: str, selected_errors: list, domain: str = None, include_error_annotations: bool = True) -> str:
    """
    Create a concise prompt for generating Java code with intentional errors.
    Enhanced to emphasize the exact number of errors required.
    
    Args:
        code_length: Length of code (short, medium, long)
        difficulty_level: Difficulty level (easy, medium, hard)
        selected_errors: List of errors to include in the code
        domain: Domain context for the code
        include_error_annotations: Whether to include error annotations
        
    Returns:
        Optimized prompt string for LLM
    """
    # Define basic code complexity by length - updated for beginners
    complexity = {
        "short": "1 simple class with 1-2 basic methods",
        "medium": "1 class with 3-5 methods of moderate complexity",
        "long": "1-2 classes with 4-8 methods and clear relationships"
    }.get(str(code_length).lower(), "1 class with methods")
    
    # Count the number of errors
    error_count = len(selected_errors)
    
    # Format errors concisely with only essential information
    error_list = []
    for error in selected_errors:
        error_type = error.get("type", "unknown").upper()
        name = error.get("name", "unknown")
        description = error.get("description", "")
        implementation_guide = error.get("implementation_guide", "")
        
        error_entry = f"{error_type} - {name}: {description}"
        if implementation_guide:
            # Include implementation guide but keep it concise
            error_entry += f"\nImplement: {implementation_guide}"
        
        error_list.append(error_entry)
    
    # Join errors with clear separation
    error_instructions = "\n\n".join(error_list)
    
    # Add difficulty-specific instructions
    beginner_focus = ""
    if difficulty_level.lower() == "easy":
        beginner_focus = """
            BEGINNER-FRIENDLY REQUIREMENTS:
            - Use very simple and descriptive variable/method names (studentName, calculateTotal)
            - Keep methods short (3-10 lines each)
            - Use basic control structures (if/else, simple loops)
            - Avoid complex nested structures
            - Make errors obvious and educational
            - Include helpful comments that explain code purpose (but not errors)
            """
    elif difficulty_level.lower() == "medium":
        beginner_focus = """
            INTERMEDIATE-LEVEL REQUIREMENTS:
            - Use a mix of simple and moderate complexity code
            - Introduce some more subtle errors that require careful reading
            - Include a variety of control structures and data types
            - Keep methods reasonably sized (5-15 lines)
            """
    else:  # hard
        beginner_focus = """
            ADVANCED-LEVEL REQUIREMENTS:
            - Create more sophisticated code structures
            - Hide errors in logical flow and edge cases
            - Use a variety of Java features and patterns
            - Challenge the student to think deeply about the code
            """
    
    # Create a focused prompt with clear role definition and beginner focus - EMPHASIZE ERROR COUNT
    prompt = f"""You are an expert Java programming instructor who creates educational code examples with specific errors for students to practice identifying and fixing.

            CRITICAL TASK:
            Generate a {code_length} Java program for a {domain or "general"} system with EXACTLY {error_count} intentional errors for code review practice. No more, no fewer.

            CRITICAL REQUIREMENTS:
            - You MUST implement EXACTLY {error_count} errors - this is NON-NEGOTIABLE
            - Only implement the SPECIFIC errors listed below - do not add any extra errors
            - Each error must be clearly marked with: // ERROR: [TYPE] - [NAME] - [Brief explanation]
            - Code should be realistic, well-structured, and match the {difficulty_level} difficulty level
            - Errors must be actual errors in the code, not just comments
            - DO NOT add comments like "// added to fix" - the errors are meant to remain as errors!
            
            {beginner_focus}

            ERRORS TO IMPLEMENT (EXACTLY {error_count} ERRORS - THIS IS CRITICAL):

            {error_instructions}

            PROVIDE TWO VERSIONS OF THE CODE:
            1. First, provide the ANNOTATED VERSION with error comments, marked with:
            ```java-annotated
            // Your code with error annotations for each of the {error_count} required errors
            ```

            2. Then, provide the CLEAN VERSION without any error comments, marked with:
            ```java-clean
            // The same code with the same {error_count} intentional errors but no error comments
            ```

            FINAL VERIFICATION:
            Before completing, verify that you have implemented EXACTLY {error_count} errors - no more, no fewer.
            """
    
    return prompt

def create_review_analysis_prompt(code: str, known_problems: list, student_review: str) -> str:
    """
    Create an optimized prompt for analyzing student code reviews.
    
    Args:
        code: The Java code being reviewed
        known_problems: List of known problems in the code
        student_review: The student's review comments
        
    Returns:
        Optimized analysis prompt
    """
    # Format known problems concisely
    problems_text = "\n".join(f"- {problem}" for problem in known_problems)
    
    # Create focused analysis prompt with educational assessment role
    prompt = f"""You are an educational assessment expert analyzing a student's Java code review. 
                Your task is to compare the student review against known issues to evaluate accuracy and completeness.

            CODE:
            ```java
            {code}
            ```

            KNOWN ISSUES IN THE CODE:
            {problems_text}

            STUDENT'S REVIEW:
            ```
            {student_review}
            ```

            Your Task:
                1. First, identify ALL actual issues in the code (not just the ones listed in "Known Issues")
                2. Determine which issues the student identified correctly
                3. List issues the student missed (ONLY COMPARE in "Known Issues")
                4. Evaluate the overall effectiveness of the student review

            JSON RESPONSE FORMAT:
            ```json
            {{
            "identified_problems": [
                {{
                "problem": "Issue description from known list",
                "student_comment": "Student's relevant comment that identified this",
                "accuracy": 0.9,
                "feedback": "Specific feedback on this identification"
                }}
            ],
            "missed_problems": [
                {{
                "problem": "Issue description from known list",
                "hint": "A helpful educational hint for finding this type of issue"
                }}
            ],
            "false_positives": [
                {{
                "student_comment": "Incorrect comment from student",
                "explanation": "Educational explanation of why this isn't an actual issue"
                }}
            ],
            "identified_count": 3,
            "total_problems": 5,
            "identified_percentage": 60.0,
            "review_sufficient": true,
            "educational_feedback": "Overall assessment of student understanding with specific improvement suggestions"
            }}
            ```
            Important Instructions:
            1. Be thorough and examine every aspect of the code
            2. Focus on logic errors, style violations, and structural problems
            3. If the student's review is incomplete, clearly state this fact
            4. Provide specific, actionable feedback that would help the student learn
            5. Be concise but complete in your analysis
            6. A review is considered "sufficient" if the student correctly identified at least 60% of the known issues.
            7. Focus on providing educationally valuable feedback that helps the student improve their code review skills.
            """
    
    return prompt

def create_feedback_prompt(code: str, known_problems: list, review_analysis: dict) -> str:
    """
    Create an optimized prompt for generating concise, focused guidance on student reviews.
    
    Args:
        code: The Java code being reviewed
        known_problems: List of known problems in the code
        review_analysis: Analysis of the student's review
        
    Returns:
        Optimized feedback prompt
    """
    # Extract data from review analysis
    identified = review_analysis.get("identified_count", 0)
    total = review_analysis.get("total_problems", len(known_problems))
    accuracy = review_analysis.get("identified_percentage", 0)
    
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
    
    # Create focused feedback prompt with educational coach role - EMPHASIZE BREVITY
    prompt = f"""You are a Java programming mentor providing concise, actionable guidance to help students improve their code review skills.

                TASK:
                Create brief, targeted feedback (maximum 3-4 sentences) for a student based on their Java code review performance.

                STUDENT PERFORMANCE SUMMARY:
                - Found {identified}/{total} issues ({accuracy:.1f}%)
                - Correctly identified: {identified_text}
                - Missed: {missed_text}

                FEEDBACK REQUIREMENTS:
                1. Be extremely concise - no more than 3-4 short sentences total
                2. Focus on 1-2 specific areas for improvement
                3. Provide concrete, actionable advice (what to look for)
                4. Use clear, direct language
                5. Be encouraging without excessive praise

                IMPORTANT: Keep your response under 100 words. Focus on brevity and clarity.
                """
    
    return prompt

def create_summary_prompt(code: str, review_history: list, final_analysis: dict) -> str:
    """
    Create a comprehensive prompt for generating final summaries.
    
    Args:
        code: The Java code being reviewed
        review_history: List of review attempts
        final_analysis: Final review analysis
        
    Returns:
        Comprehensive summary prompt
    """
    # Extract final performance metrics
    identified = final_analysis.get("identified_count", 0)
    total = final_analysis.get("total_problems", 0)
    accuracy = final_analysis.get("identified_percentage", 0)
    
    # Format review iterations
    iterations = len(review_history)
    iterations_text = ""
    
    for i, review in enumerate(review_history, 1):
        analysis = review.get("review_analysis", {})
        identified_count = analysis.get("identified_count", 0)
        identified_pct = analysis.get("identified_percentage", 0)
        
        iterations_text += f"Attempt {i}: Found {identified_count}/{total} issues ({identified_pct:.1f}%)\n"
    
    # Create comprehensive summary prompt
    prompt = f"""You are an educational assessment specialist who creates comprehensive learning summaries for code review practice.

            TASK:
            Create a detailed educational summary of this student's code review practice session.

            CODE REVIEWED:
            ```java
            {code}
            ```

            PERFORMANCE SUMMARY:
            - Final score: {identified}/{total} issues identified ({accuracy:.1f}%)
            - Number of review attempts: {iterations}
            - Progress across attempts:
            {iterations_text}

            SUMMARY REQUIREMENTS:
            1. Create a comprehensive, educational summary that helps the student learn from this exercise
            2. Focus on skill development and progress across attempts (if multiple)
            3. Highlight both strengths and areas for improvement
            4. Include specific code examples from their review
            5. Provide actionable recommendations for continued learning
            6. Use markdown formatting for readability

            SUMMARY STRUCTURE:
            - Overall Performance Assessment
            - Skills Demonstrated (with specific examples)
            - Learning Opportunities (what they can improve)
            - Progress Analysis (how they improved across attempts)
            - Practical Recommendations (specific tips and resources)
            - Next Steps for Continued Learning

            Make the summary educational, encouraging, and focused on transferable skills.
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

def get_error_count_for_difficulty(difficulty: str) -> int:
    """
    Get appropriate error count based on difficulty level.
    
    Args:
        difficulty: Difficulty level (easy, medium, hard)
        
    Returns:
        Number of errors to include
    """
    difficulty_map = {
        "easy": 2,
        "medium": 4,
        "hard": 6
    }
    return difficulty_map.get(str(difficulty).lower(), 4)

def generate_comparison_report(evaluation_errors: List[str], review_analysis: Dict[str, Any]) -> str:
    """
    Generate a comparison report between student review and evaluated errors.
    
    Args:
        evaluation_errors: List of errors found by the evaluation
        review_analysis: Analysis of the student review
        
    Returns:
        Formatted comparison report
    """
    # Create report header
    report = "# Detailed Comparison: Your Review vs. Actual Issues\n\n"
    
    # New section explaining the comparison method
    report += "## How Reviews Are Compared\n\n"
    report += "Your review is compared to the known problems in the code using a semantic matching approach. "
    report += "This means we look for whether you've identified the key aspects of each issue, rather than requiring exact matching phrases.\n\n"
    
    report += "For each issue, the system checks if your comments include:\n"
    report += "1. **The correct location** of the error (line numbers, method names, etc.)\n"
    report += "2. **The appropriate error type or category** (e.g., NullPointerException, naming convention)\n"
    report += "3. **A clear explanation** of why it's problematic\n\n"
    
    report += "A problem is considered 'identified' if you correctly mentioned its key aspects. "
    report += "Partial credit may be given for partially identified issues. "
    report += "False positives are issues you reported that don't match any actual problems in the code.\n\n"
    
    # Problems section
    report += "## Code Issues Analysis\n\n"
    
    # Safely extract data from review analysis
    identified_problems = review_analysis.get("identified_problems", [])
    missed_problems = review_analysis.get("missed_problems", [])
    false_positives = review_analysis.get("false_positives", [])
    
    # IMPROVED: Get the total problems count directly from the review analysis
    # This ensures we're using the same count everywhere
    total_problems = review_analysis.get("total_problems", 0)
    
    # If total_problems is not available, try original_error_count
    if total_problems <= 0:
        total_problems = review_analysis.get("original_error_count", 0)
        
    # Last resort fallback to evaluation_errors length
    if total_problems <= 0:
        total_problems = len(evaluation_errors)
    
    # Ensure all problems are properly converted to strings
    known_problems_str = [str(p) if not isinstance(p, str) else p for p in evaluation_errors]
    identified_problems_str = [str(p) if not isinstance(p, str) else p for p in identified_problems]
    missed_problems_str = [str(p) if not isinstance(p, str) else p for p in missed_problems]
    false_positives_str = [str(p) if not isinstance(p, str) else p for p in false_positives]
    
    # Issues found correctly
    if identified_problems_str:
        report += "### Issues You Identified Correctly\n\n"
        for i, problem in enumerate(identified_problems_str, 1):
            report += f"**{i}. {problem}**\n\n"
            report += "Great job finding this issue! "
            report += "This demonstrates your understanding of this type of problem.\n\n"
    
    # Issues missed with detailed guidance
    if missed_problems_str:
        report += "### Issues You Missed\n\n"
        for i, problem in enumerate(missed_problems_str, 1):
            report += f"**{i}. {problem}**\n\n"
            
            # Enhanced guidance with example comment format
            problem_lower = problem.lower()
            report += "**How to identify this issue:**\n\n"
            
            if "null" in problem_lower or "nullpointer" in problem_lower:
                report += "When reviewing Java code, look for variables that might be null before being accessed. "
                report += "Check for null checks before method calls or field access. Missing null checks often lead to NullPointerExceptions at runtime.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [NullPointerException Risk] - The variable 'name' is accessed without a null check, which could cause a runtime exception`\n\n"
            elif "naming" in problem_lower or "convention" in problem_lower:
                report += "Check that class names use UpperCamelCase, while methods and variables use lowerCamelCase. "
                report += "Constants should use UPPER_SNAKE_CASE. Consistent naming improves code readability and maintainability.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Naming Convention] - The variable 'user_name' should use lowerCamelCase format (userName)`\n\n"
            elif "equal" in problem_lower or "==" in problem_lower:
                report += "String and object comparisons should use the .equals() method instead of the == operator, which only compares references. "
                report += "Using == for content comparison is a common error that can lead to unexpected behavior.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Object Comparison] - String comparison uses == operator instead of .equals() method`\n\n"
            elif "array" in problem_lower or "index" in problem_lower:
                report += "Always verify that array indices are within valid ranges before accessing elements. "
                report += "Check for potential ArrayIndexOutOfBoundsException risks, especially in loops.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Array Bounds] - Array access without bounds checking could cause ArrayIndexOutOfBoundsException`\n\n"
            elif "whitespace" in problem_lower or "indent" in problem_lower:
                report += "Look for consistent indentation and proper whitespace around operators and keywords. "
                report += "Proper formatting makes code more readable and maintainable.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Formatting] - Inconsistent indentation makes the code hard to read`\n\n"
            else:
                report += "When identifying issues, be specific about the location, type of error, and why it's problematic. "
                report += "Include line numbers and detailed explanations in your comments.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Error Type] - Description of the issue and why it's problematic`\n\n"
    
    # False positives
    if false_positives_str:
        report += "### Issues You Incorrectly Identified\n\n"
        for i, problem in enumerate(false_positives_str, 1):
            report += f"**{i}. {problem}**\n\n"
            report += "This wasn't actually an issue in the code. "
            report += "Be careful not to flag correct code as problematic.\n\n"
    
    # Calculate metrics consistently using total_problems
    identified_count = len(identified_problems_str)
    missed_count = len(missed_problems_str)
    false_positive_count = len(false_positives_str)
    
    accuracy = (identified_count / total_problems * 100) if total_problems > 0 else 0
    
    # Overall assessment
    report += "### Overall Assessment\n\n"
    
    if accuracy >= 80:
        report += "**Excellent review!** You found most of the issues in the code.\n\n"
    elif accuracy >= 60:
        report += "**Good review.** You found many issues, but missed some important ones.\n\n"
    elif accuracy >= 40:
        report += "**Fair review.** You found some issues, but missed many important ones.\n\n"
    else:
        report += "**Needs improvement.** You missed most of the issues in the code.\n\n"
    
    report += f"- You identified {identified_count} out of {total_problems} issues ({accuracy:.1f}%)\n"
    report += f"- You missed {total_problems - identified_count} issues\n"
    report += f"- You incorrectly identified {false_positive_count} non-issues\n\n"
    
    # Add improvement tips
    report += "## Tips for Improvement\n\n"
    
    if missed_problems_str:
        # Categories of missed issues
        missed_categories = []
        
        for problem in missed_problems_str:
            problem_lower = problem.lower()
            if "null" in problem_lower:
                missed_categories.append("null pointer handling")
            elif "naming" in problem_lower or "convention" in problem_lower:
                missed_categories.append("naming conventions")
            elif "javadoc" in problem_lower or "comment" in problem_lower:
                missed_categories.append("documentation")
            elif "exception" in problem_lower or "throw" in problem_lower:
                missed_categories.append("exception handling")
            elif "loop" in problem_lower or "condition" in problem_lower:
                missed_categories.append("logical conditions")
            elif "whitespace" in problem_lower or "indentation" in problem_lower:
                missed_categories.append("code formatting")
            elif "array" in problem_lower or "index" in problem_lower:
                missed_categories.append("array handling")
        
        # Remove duplicates and sort
        missed_categories = sorted(set(missed_categories))
        
        if missed_categories:
            report += "Based on your review, focus on these areas in future code reviews:\n\n"
            for category in missed_categories:
                report += f"- **{category.title()}**\n"
            report += "\n"
    
    # Add systematic approach suggestion
    report += """### Systematic Review Approach

        For more thorough code reviews, try this systematic approach:

        1. **First pass**: Check for syntax errors, compilation issues, and obvious bugs
        2. **Second pass**: Examine naming conventions, code style, and documentation
        3. **Third pass**: Analyze logical flow, edge cases, and potential runtime errors
        4. **Final pass**: Look for performance issues, security concerns, and maintainability problems

        By following a structured approach, you'll catch more issues and provide more comprehensive reviews.
        """
    
    # Add effective comment format
    report += """
        ### Effective Comment Format

        When writing code review comments, use this format for clarity and consistency:

        ```
        Line X: [Error Type] - Description of the issue and why it's problematic
        ```

        For example:
        ```
        Line 42: [NullPointerException Risk] - The 'user' variable could be null here, add a null check before calling methods
        ```

        This format helps others quickly understand the location, type, and impact of each issue.
        """
    
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