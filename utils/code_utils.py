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

def format_list(items: List[str]) -> str:
    """
    Format a list of items as a bullet list.
    
    Args:
        items: List of string items
        
    Returns:
        Formatted bullet list as string
    """
    return "\n".join([f"- {item}" for item in items])

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
    
    Args:
        code_length: Length of code (short, medium, long)
        difficulty_level: Difficulty level (easy, medium, hard)
        selected_errors: List of errors to include in the code
        domain: Domain context for the code
        include_error_annotations: Whether to include error annotations
        
    Returns:
        Optimized prompt string for LLM
    """
    # Define basic code complexity by length
    complexity = {
        "short": "1 class with 2-4 methods",
        "medium": "1-2 classes with 4-6 methods",
        "long": "2-3 classes with proper relationships"
    }.get(str(code_length).lower(), "1-2 classes")
    
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
    
    # Create a focused prompt requesting both annotated and clean versions
    # Create a focused prompt with clear role definition
    prompt = f"""You are an expert Java programming instructor who creates educational code examples with specific errors for students to practice identifying and fixing.

            TASK:
            Generate a {code_length} Java program for a {domain or "general"} system with intentional errors for code review practice.

            REQUIREMENTS:
            - Make a {complexity} Java program that implements ALL these errors
            - Each error must be clearly marked with: // ERROR: [TYPE] - [NAME] - [Brief explanation]
            - Code should be realistic and well-structured
            - Return TWO VERSIONS of the code as described below

            ERRORS TO IMPLEMENT:

            {error_instructions}

            PROVIDE TWO VERSIONS OF THE CODE:
            1. First, provide the ANNOTATED VERSION with error comments, marked with:
            ```java-annotated
            // Your code with error annotations
            ```

            2. Then, provide the CLEAN VERSION without any error comments, marked with:
            ```java-clean
            // The same code without any error comments
            ```
            """
    
    return prompt

def create_evaluation_prompt(code: str, requested_errors: list) -> str:
    """
    Create a concise prompt for evaluating whether code contains required errors.
    
    Args:
        code: The generated Java code
        requested_errors: List of errors that should be implemented
        
    Returns:
        Optimized evaluation prompt
    """
    # Format requested errors concisely
    error_list = []
    for error in requested_errors:
        error_type = error.get("type", "").upper()
        name = error.get("name", "")
        error_list.append(f"{error_type} - {name}")
    
    error_instructions = "\n".join(error_list)
    
     # Create focused evaluation prompt with clear role definition
    prompt = f"""You are a Java code assessment expert who specializes in identifying specific error patterns in code examples for educational purposes.

            TASK:
            Evaluate this Java code and determine if it correctly implements all requested errors.

            CODE:
            ```java
            {code}
            ```

            REQUESTED ERRORS:
            {error_instructions}

            EVALUATION REQUIREMENTS:
            1. For each error, determine if it's correctly implemented in the code
            2. Identify the exact line number and code segment where each error is found
            3. For missing errors, explain why you couldn't find them
            4. Return a detailed analysis in JSON format

            JSON RESPONSE FORMAT:
            ```json
            {{
            "found_errors": [
                {{
                "error_type": "BUILD",
                "error_name": "NullPointerException",
                "line_number": 42,
                "code_segment": "String str = null; int length = str.length();",
                "explanation": "This implements a NullPointerException by calling length() on a null String"
                }}
            ],
            "missing_errors": [
                {{
                "error_type": "CHECKSTYLE",
                "error_name": "MemberName",
                "explanation": "No variable names breaking member naming conventions found in the code"
                }}
            ],
            "valid": false,
            "feedback": "The code implements 4 out of 5 requested errors. Missing the CHECKSTYLE-MemberName error."
            }}
            ```

            The "valid" field should be true only if ALL requested errors are properly implemented.
            """
    
    return prompt

def create_regeneration_prompt(code: str, domain: str, missing_errors: list, found_errors: list, requested_errors: list) -> str:
    """
    Create a focused prompt for regenerating code with missing errors.
    
    Args:
        code: The original code to improve
        domain: Domain of the code (must be consistent with original)
        missing_errors: List of error keys that need to be implemented
        found_errors: List of error keys already implemented correctly
        requested_errors: Full list of requested error dictionaries
        
    Returns:
        Optimized regeneration prompt
    """
    # Create detailed instructions for missing errors
    missing_instructions = []
    for error_key in missing_errors:
        # Find the full error details
        for error in requested_errors:
            if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                error_type = error.get("type", "").upper()
                name = error.get("name", "")
                guide = error.get("implementation_guide", "")
                
                instruction = f"{error_type} - {name}"
                if guide:
                    instruction += f": {guide}"
                missing_instructions.append(instruction)
                break
    
    missing_text = "\n".join(f"- {instr}" for instr in missing_instructions)
    found_text = "\n".join(f"- {err}" for err in found_errors)
    
    # Create a focused prompt with expert role definition
    prompt = f"""You are a code revision specialist who can expertly modify Java code to incorporate specific error patterns while preserving existing structure and functionality.

            TASK:
            Improve this Java code by adding the MISSING errors while preserving the EXISTING errors.

            ORIGINAL CODE DOMAIN: {domain}

            MISSING ERRORS - Add these to the code:
            {missing_text}

            CORRECTLY IMPLEMENTED - Keep these in your solution:
            {found_text}

            REQUIREMENTS:
            1. Use EXACTLY the same {domain} domain and maintain the original code structure
            2. Add ALL the missing errors in a natural way that makes sense in the code context
            3. Preserve all correctly implemented errors (don't modify their implementation)
            4. Mark EACH error with a comment: // ERROR: [TYPE] - [NAME] - [Brief explanation]
            5. Return both annotated and clean versions of the revised code

            PROVIDE TWO VERSIONS OF THE CODE:
            1. First, provide the ANNOTATED VERSION with error comments, marked with:
            ```java-annotated
            // Your code with error annotations
            ```

            2. Then, provide the CLEAN VERSION without any error comments, marked with:
            ```java-clean
            // The same code without any error comments
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
                Your task is to compare their review against known issues to evaluate accuracy and completeness.

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
                3.List issues the student missed
                4. Evaluate the overall effectiveness of their review

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
    Create an optimized prompt for generating feedback on student reviews.
    
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
            student_comment = problem.get("student_comment", "")
            identified_text += f"- Problem: {problem_text}\n  Comment: {student_comment}\n"
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
    prompt = f"""You are an expert Java programming mentor who provides constructive educational feedback to help students improve their code review skills.

                TASK:
                Create personalized, educational feedback for a student based on their Java code review performance.

                STUDENT PERFORMANCE SUMMARY:
                - Found {identified}/{total} issues ({accuracy:.1f}%)
                - Correctly identified issues:\n{identified_text}
                - Missed issues:\n{missed_text}

                FEEDBACK REQUIREMENTS:
                1. Be encouraging and constructive - focus on growth and learning
                2. Highlight specific strengths in their review
                3. Provide targeted educational guidance for missed issues
                4. Include practical tips for improving code review skills
                5. Suggest 1-2 specific resources for further learning
                6. Use clear, concise language with markdown formatting

                FEEDBACK STRUCTURE:
                - Start with a positive assessment of their effort
                - Include specific examples from their review to illustrate strengths
                - For missed issues, explain educational patterns rather than just listing what they missed
                - Focus on transferable skills they can apply to future code reviews
                - End with specific next steps for improvement

                Use markdown formatting for readability and structure.
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

def extract_code_from_response(response, tag: str = None) -> str:
    """
    Extract Java code from LLM response with optional tag filtering.
    Handles both string responses and AIMessage/ChatMessage objects.
    
    Args:
        response: Text response from LLM or AIMessage/ChatMessage object
        tag: Optional tag to look for (e.g., "java-annotated", "java-clean")
        
    Returns:
        Extracted Java code
    """
    # Check for None or empty response
    if not response:
        return ""
    
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
    
    # If a specific tag is provided, look for that tag
    if tag:
        tag_pattern = f"```{tag}\\s*(.*?)\\s*```"
        code_blocks = re.findall(tag_pattern, response_text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
    
    # Try to extract code from any code blocks
    code_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', response_text, re.DOTALL)
    
    if code_blocks:
        # Return the largest code block
        return max(code_blocks, key=len)
    
    # If no code blocks are found, assume the entire response is code
    return response_text.strip()

def extract_both_code_versions(response) -> Tuple[str, str]:
    """
    Extract both annotated and clean code versions from LLM response.
    Handles both string responses and AIMessage/ChatMessage objects.
    
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
    
    # If we have annotated code but no clean code, we'll need to handle that at the calling site
    # by using strip_error_annotations_with_llm
    
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

def generate_comparison_report(known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
    """
    Generate a comparison report between student review and known problems.
    
    Args:
        known_problems: List of known problems in the code
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
    
    # Ensure all problems are properly converted to strings
    known_problems_str = [str(p) if not isinstance(p, str) else p for p in known_problems]
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
    
    # Calculate some metrics
    total_problems = len(known_problems_str)
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
    report += f"- You missed {missed_count} issues\n"
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

def strip_error_annotations(code: str) -> str:
    """
    Remove error annotation comments from code while preserving string literals.
    Focused on standardized format: // ERROR: [TYPE] - [NAME] - [Description]
    but still handles legacy formats for backward compatibility.
    Preserves comments that start with "// Intentional error:" as these indicate
    deliberate errors for educational purposes.
    
    Args:
        code: The code with error annotations
        
    Returns:
        Clean code without error annotations (except for "// Intentional error:" comments)
    """
    import re
    
    # Skip processing if code is empty
    if not code:
        return code
    
    # Standard format pattern (should match most generated annotations)
    standard_pattern = r'^\s*//\s*ERROR:\s*\[.*?\].*\n'
    
    # Match and remove specific error annotation patterns
    error_patterns = [
        # Primary standardized format
        r'^\s*//\s*ERROR:\s*.*-.*\n',
        
        # Legacy formats for backward compatibility
        r'^\s*//\s*ERROR TYPE:.*\n',
        r'^\s*//\s*ERROR NAME:.*\n',
        r'^\s*//\s*TODO: Fix.*\n',
        r'^\s*//\s*FIXME:.*\n',
        
        # Very specific patterns that indicate error annotations
        r'^\s*//.*BUILD ERROR.*\n',
        r'^\s*//.*CHECKSTYLE ERROR.*\n',
        r'^\s*//.*LOGICAL ERROR.*\n',
        r'^\s*//.*RUNTIME ERROR.*\n',
        
        # General case for problem area annotations
        r'^\s*//\s*Problem area:.*\n',
    ]
    
    # Process the code line by line to preserve "// Intentional error:" comments
    lines = code.splitlines()
    result_lines = []
    
    for line in lines:
        # Skip lines that match our error patterns unless they start with "// Intentional error:"
        if re.match(r'^\s*//\s*Intentional error:', line):
            # Keep this line - it's an intentional error marker we want to preserve
            result_lines.append(line)
            continue
            
        # Check if this line matches any of our error annotation patterns
        is_error_annotation = False
        
        # Check against standard pattern
        if re.match(standard_pattern, line):
            is_error_annotation = True
        else:
            # Check against other patterns
            for pattern in error_patterns:
                if re.match(pattern, line):
                    is_error_annotation = True
                    break
        
        # Keep the line if it's not an error annotation
        if not is_error_annotation:
            result_lines.append(line)
    
    # Join the lines back together
    result = '\n'.join(result_lines)
    
    # Clean up any empty lines created by removing annotations
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result

def strip_error_annotations_with_llm(code: str, llm: BaseLanguageModel = None) -> str:
    """
    Use LLM to remove error annotation comments from Java code.
    
    Args:
        code: The Java code with error annotations
        llm: Language model to use for cleaning
        
    Returns:
        Clean code without error annotations
    """
    if not code or not llm:
        return code
        
    prompt = f"""Remove all error annotation comments from this Java code.
    
            Keep the actual Java code unchanged, but remove any comments that describe errors.
            Specifically, remove all lines that:
            1. Start with // ERROR:
            2. Start with // TODO: Fix
            3. Contain phrases like "intentional error" or "error annotation"
            4. Describe bugs, issues, or errors in comments

            Do NOT change any actual Java code - only remove the comment lines.
            Return only the cleaned Java code without any explanation.

            Original code:
            ```java
            {code}
            """
    try:
        # Use the LLM to clean the code
        response = llm.invoke(prompt)
        
        # Extract the code from the response
        clean_code = extract_code_from_response(response)
        
        # If extraction failed, return the full response (which should be just the code)
        if not clean_code:
            # Remove any markdown formatting that might be present
            clean_code = re.sub(r'```.*\n|```', '', response).strip()
        
        return clean_code
        
    except Exception as e:
        logger.error(f"Error using LLM to strip error annotations: {str(e)}")
        # Fall back to the regex-based method as a backup
        #return strip_error_annotations(code)

def process_llm_response(response):
    """
    Process LLM response to handle different formats from different providers.
    Cleans up formatting issues, especially with Groq responses.
    
    Args:
        response: Response from LLM (string, AIMessage, or dict)
        
    Returns:
        Cleaned string content
    """
    # Handle None case
    if response is None:
        return ""
    
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