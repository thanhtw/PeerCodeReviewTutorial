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

def create_regeneration_prompt(code: str, domain: str, missing_errors: list, found_errors: list, requested_errors: list) -> str:
    """
    Create a focused prompt for regenerating code with missing errors and removing extra errors.
    Enhanced to provide clear instructions for exact error requirements.
    
    Args:
        code: The original code to improve
        domain: Domain of the code (must be consistent with original)
        missing_errors: List of error keys that need to be implemented
        found_errors: List of error keys already implemented correctly
        requested_errors: Full list of requested error dictionaries      
        
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
    
     # Use provided domain or default to "general"
    
    domain_str = domain or "general"
    # Create a focused prompt with clear role definition and beginner focus - EMPHASIZE ERROR COUNT
    prompt = f"""You are an expert Java programming instructor who creates educational code examples with specific errors for students to practice identifying and fixing.

            CRITICAL TASK:
            Generate a {code_length} Java program for a {domain_str} system with EXACTLY {error_count} intentional errors for code review practice. No more, no fewer.

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

def generate_comparison_report(evaluation_errors: List[str], review_analysis: Dict[str, Any], 
                              review_history: List[Dict[str, Any]] = None) -> str:
    """
    Generate a concise comparison report showing progress across review attempts.
    
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