"""
Feedback Manager module for Java Peer Review Training System.

This module provides the FeedbackManager class which coordinates
the feedback loop between student reviews and AI-generated feedback.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
from .student_response_evaluator import StudentResponseEvaluator
from langchain_core.language_models import BaseLanguageModel
from utils.code_utils import process_llm_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewIteration:
    """Class for storing a single review iteration."""
    
    def __init__(self, iteration_number: int, student_review: str, 
                 review_analysis: Dict[str, Any], targeted_guidance: Optional[str] = None, 
                 timestamp: Optional[str] = None):
        """
        Initialize a review iteration record.
        
        Args:
            iteration_number: The sequence number of this review iteration
            student_review: The student's review comments
            review_analysis: Analysis of the student review
            targeted_guidance: Guidance provided for the next iteration
            timestamp: Timestamp of this review iteration
        """
        self.iteration_number = iteration_number
        self.student_review = student_review
        self.review_analysis = review_analysis
        self.targeted_guidance = targeted_guidance
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "student_review": self.student_review,
            "review_analysis": self.review_analysis,
            "targeted_guidance": self.targeted_guidance,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewIteration':
        """Create a ReviewIteration from a dictionary."""
        return cls(
            iteration_number=data.get("iteration_number", 1),
            student_review=data.get("student_review", ""),
            review_analysis=data.get("review_analysis", {}),
            targeted_guidance=data.get("targeted_guidance"),
            timestamp=data.get("timestamp")
        )

class FeedbackManager:
    """
    Manages the feedback loop between student reviews and AI-generated feedback.
    
    This class coordinates the iterative process of student review, analysis,
    and feedback generation, maintaining the history of review iterations.
    """
    
    def __init__(self, evaluator: StudentResponseEvaluator,
                max_iterations: int = 3):
        """
        Initialize the FeedbackManager.
        
        Args:
            evaluator: StudentResponseEvaluator for analyzing reviews
            max_iterations: Maximum number of review iterations
        """
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.review_history = []
        self.current_iteration = 1
        self.review_sufficient = False
        self.code_snippet = ""
        self.known_problems = []
    
    def start_new_review_session(self, code_snippet: str, evaluation_result: Dict[str, Any]):
        """
        Start a new review session.
        
        Args:
            code_snippet: The code snippet to be reviewed
            evaluation_result: Evaluation results containing found errors
        """
        self.code_snippet = code_snippet
        # Extract found problems from evaluation result
        self.known_problems = evaluation_result.get('found_errors', [])
        self.review_history = []
        self.current_iteration = 1
        self.review_sufficient = False
        logger.info("Started new review session")
    
    def submit_review(self, student_review: str) -> Dict[str, Any]:
        """
        Submit a student review for analysis.
        
        Args:
            student_review: The student's review comments
            
        Returns:
            Dictionary with analysis results and guidance
        """
        if not self.code_snippet or not self.known_problems:
            logger.error("Cannot submit review without active code snippet")
            return {
                "error": "No active code snippet to review"
            }
        
        # Analyze the student review
        logger.info(f"Analyzing student review (iteration {self.current_iteration})")
        review_analysis = self.evaluator.evaluate_review(
            code_snippet=self.code_snippet,
            known_problems=self.known_problems,
            student_review=student_review
        )
        
        # Check if the review is sufficient
        self.review_sufficient = review_analysis.get("review_sufficient", False)
        
        # Generate targeted guidance if needed
        targeted_guidance = None
        if not self.review_sufficient and self.current_iteration < self.max_iterations:
            logger.info(f"Generating guidance for iteration {self.current_iteration}")
            targeted_guidance = self.evaluator.generate_targeted_guidance(
                code_snippet=self.code_snippet,
                known_problems=self.known_problems,
                student_review=student_review,
                review_analysis=review_analysis,
                iteration_count=self.current_iteration,
                max_iterations=self.max_iterations
            )
        
        # Create a review iteration record
        current_iteration = ReviewIteration(
            iteration_number=self.current_iteration,
            student_review=student_review,
            review_analysis=review_analysis,
            targeted_guidance=targeted_guidance,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Add to review history
        self.review_history.append(current_iteration)
        
        # Prepare the result
        result = {
            "iteration_count": self.current_iteration,
            "review_analysis": review_analysis,
            "review_sufficient": self.review_sufficient,
            "targeted_guidance": targeted_guidance,
            "next_steps": "summarize" if self.review_sufficient or self.current_iteration >= self.max_iterations else "iterate"
        }
        
        # Increment iteration count for next review
        self.current_iteration += 1
        
        return result
    
    def get_review_history(self) -> List[Dict[str, Any]]:
        """
        Get the review history.
        
        Returns:
            List of review iteration dictionaries
        """
        return [iteration.to_dict() for iteration in self.review_history]
    
    def get_latest_review(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest review iteration.
        
        Returns:
            Dictionary of the latest review iteration or None if no reviews
        """
        if not self.review_history:
            return None
        
        return self.review_history[-1].to_dict()
    
    def generate_final_feedback(self,llm: Optional[BaseLanguageModel] = None,include_resources: bool = True,include_visualizations: bool = True) -> str:
        """
        Generate enhanced final feedback with educational resources and data visualizations.
        
        Args:
            llm: Optional language model for generating customized feedback
            include_resources: Whether to include educational resources
            include_visualizations: Whether to include performance visualizations
            
        Returns:
            Enhanced feedback text in markdown format
        """
        if not self.review_history:
            return "No reviews submitted yet."
        
        # If LLM is available, use it for generating personalized feedback
        if llm:
            return self._generate_llm_feedback(llm)
        
        # Otherwise, use enhanced template-based approach
        return self._generate_template_feedback(include_resources, include_visualizations)
    
    def _generate_llm_feedback(self, llm: BaseLanguageModel) -> str:
        """
        Generate personalized feedback using an LLM.
        
        Args:
            llm: Language model for generating feedback
            
        Returns:
            Personalized feedback text
        """
        try:
            # Get the code snippet and known problems
            code_snippet = self.code_snippet
            known_problems = self.known_problems
            
            # Get all reviews and analyses
            reviews = []
            analyses = []
            for iteration in self.review_history:
                reviews.append(iteration.student_review)
                analyses.append(iteration.review_analysis)
            
            # Create a prompt for the LLM
            system_prompt = """You are an expert Java programming mentor who provides comprehensive educational feedback on code reviews. Your feedback should be:

                        1. Educational - highlight learning opportunities and patterns
                        2. Specific - address the student's exact performance
                        3. Constructive - offer actionable advice for improvement
                        4. Progressive - focus on skill development over time
                        5. Balanced - acknowledge strengths while addressing weaknesses

                        Use markdown formatting for clear, professional-looking feedback."""
            
            # Format the reviews and analyses for the prompt
            reviews_text = []
            for i, (review, analysis) in enumerate(zip(reviews, analyses), 1):
                reviews_text.append(f"REVIEW ATTEMPT {i}:\n```\n{review}\n```\n")
                
                # Add key metrics for this review
                identified = analysis.get("identified_count", 0)
                total = analysis.get("total_problems", len(known_problems))
                percentage = analysis.get("identified_percentage", 0)
                reviews_text.append(f"- Found {identified}/{total} issues ({percentage:.1f}%)\n")
            
            all_reviews = "\n".join(reviews_text)
            
            # Get final review analysis
            final_analysis = analyses[-1] if analyses else {}
            
            # Extract problem lists
            identified_problems = []
            if "identified_problems" in final_analysis:
                if isinstance(final_analysis["identified_problems"], list):
                    for problem in final_analysis["identified_problems"]:
                        if isinstance(problem, dict) and "problem" in problem:
                            identified_problems.append(problem["problem"])
                        elif isinstance(problem, str):
                            identified_problems.append(problem)
            
            missed_problems = []
            if "missed_problems" in final_analysis:
                if isinstance(final_analysis["missed_problems"], list):
                    for problem in final_analysis["missed_problems"]:
                        if isinstance(problem, dict) and "problem" in problem:
                            missed_problems.append(problem["problem"])
                        elif isinstance(problem, str):
                            missed_problems.append(problem)
            
            false_positives = []
            if "false_positives" in final_analysis:
                if isinstance(final_analysis["false_positives"], list):
                    for item in final_analysis["false_positives"]:
                        if isinstance(item, dict) and "student_comment" in item:
                            false_positives.append(item["student_comment"])
                        elif isinstance(item, str):
                            false_positives.append(item)
            
            # Format the problems for the prompt
            identified_text = "\n".join([f"- {p}" for p in identified_problems])
            missed_text = "\n".join([f"- {p}" for p in missed_problems])
            false_positive_text = "\n".join([f"- {p}" for p in false_positives])
            
            prompt = f"""
                Create comprehensive educational feedback for a student who completed a Java code review exercise.

                CODE SNIPPET:
                ```java
                {code_snippet}
                ```

                KNOWN ISSUES IN THE CODE:
                {", ".join(known_problems)}

                STUDENT'S REVIEW HISTORY:
                {all_reviews}

                FINAL PERFORMANCE:
                - Correctly identified: {identified_problems and identified_text or "None"}
                - Missed issues: {missed_problems and missed_text or "None"}
                - False positives: {false_positives and false_positive_text or "None"}

                INSTRUCTIONS:
                1. Create complete educational feedback with these sections:
                - Review Performance (quantitative assessment with metrics)
                - Strengths (what they did well)
                - Areas for Improvement (specific skills to develop)
                - Missed Issues Analysis (explanation of what they missed and why it matters)
                - False Positives Analysis (if any)
                - Progress Analysis (if multiple attempts)
                - Learning Resources (recommended readings or exercises)
                - Next Steps (concrete suggestions for continued improvement)

                2. Include specific examples from their reviews
                3. Use markdown formatting for readability
                4. Keep a constructive, encouraging tone
                5. Provide educational value beyond this specific exercise

                The feedback should help them become better at code review in general, not just finding the specific issues in this exercise.
                """
            
            # Generate feedback
            logger.info("Generating feedback with LLM")
            response = llm.invoke(system_prompt + "\n\n" + prompt)
            # Process the response to fix formatting issues
            processed_response = process_llm_response(response)
            
            # Return the generated feedback
            return processed_response
            
        except Exception as e:
            logger.error(f"Error generating LLM feedback: {str(e)}")
            # Fall back to template-based feedback
            return self._generate_template_feedback()

    def _generate_template_feedback(self, include_resources: bool = True, include_visualizations: bool = True) -> str:
        """
        Generate feedback using an enhanced template-based approach.
        
        Args:
            include_resources: Whether to include educational resources
            include_visualizations: Whether to include performance visualizations
            
        Returns:
            Template-based feedback text
        """
        # Get the final/best review
        latest_review = self.review_history[-1]
        
        # Build a feedback summary
        feedback = "# Final Review Feedback\n\n"
        
        # Analysis stats
        analysis = latest_review.review_analysis
        identified_count = analysis.get("identified_count", 0)
        total_problems = analysis.get("total_problems", len(self.known_problems))
        identified_percentage = analysis.get("identified_percentage", 0)
        
        # Performance summary
        feedback += "## Review Performance\n\n"
        feedback += f"You identified {identified_count} out of {total_problems} issues "
        feedback += f"({identified_percentage:.1f}% accuracy).\n\n"
        
        if include_visualizations:
            feedback += "### Performance Visualization\n\n"
            feedback += "```\n"
            # Create a simple ASCII chart
            chart_width = 40
            filled_chars = int((identified_percentage / 100) * chart_width)
            empty_chars = chart_width - filled_chars
            feedback += f"Issues Found: [{filled_chars * '█'}{empty_chars * '░'}] {identified_percentage:.1f}%\n"
            feedback += "```\n\n"
        
        # Strengths and areas for improvement
        feedback += "## Strengths\n\n"
        
        # Extract identified problems for display
        identified_problems = []
        if "identified_problems" in analysis:
            if isinstance(analysis["identified_problems"], list):
                for problem in analysis["identified_problems"]:
                    if isinstance(problem, dict) and "problem" in problem:
                        identified_problems.append(problem["problem"])
                    elif isinstance(problem, str):
                        identified_problems.append(problem)
        else:
            identified_problems = analysis.get("identified_problems", [])
        
        if identified_problems:
            feedback += "You correctly identified:\n\n"
            for problem in identified_problems:
                feedback += f"- {problem}\n"
        else:
            feedback += "Keep practicing to improve your error identification skills.\n"
        
        feedback += "\n## Areas for Improvement\n\n"
        
        # Extract missed problems for display
        missed_problems = []
        if "missed_problems" in analysis:
            if isinstance(analysis["missed_problems"], list):
                for problem in analysis["missed_problems"]:
                    if isinstance(problem, dict) and "problem" in problem:
                        missed_problems.append(problem["problem"])
                        # Include hint if available
                        if "hint" in problem:
                            missed_problems[-1] += f" (Hint: {problem['hint']})"
                    elif isinstance(problem, str):
                        missed_problems.append(problem)
        else:
            missed_problems = analysis.get("missed_problems", [])
        
        if missed_problems:
            feedback += "You missed these important issues:\n\n"
            for problem in missed_problems:
                feedback += f"- {problem}\n"
        else:
            feedback += "Great job! You found all the issues.\n"
        
        # False positives section
        false_positives = []
        if "false_positives" in analysis:
            if isinstance(analysis["false_positives"], list):
                for item in analysis["false_positives"]:
                    if isinstance(item, dict) and "student_comment" in item:
                        comment = item["student_comment"]
                        explanation = item.get("explanation", "")
                        false_positives.append(f"{comment}" + (f" ({explanation})" if explanation else ""))
                    elif isinstance(item, str):
                        false_positives.append(item)
        else:
            false_positives = analysis.get("false_positives", [])
        
        if false_positives:
            feedback += "\n## False Positives\n\n"
            feedback += "These aspects you identified weren't actual issues:\n\n"
            for issue in false_positives:
                feedback += f"- {issue}\n"
        
        # Progress across iterations
        if len(self.review_history) > 1:
            feedback += "\n## Progress Across Iterations\n\n"
            
            # Create a progress table
            feedback += "| Iteration | Issues Found | Accuracy |\n"
            feedback += "|-----------|--------------|----------|\n"
            
            for iteration in self.review_history:
                iter_num = iteration.iteration_number
                iter_analysis = iteration.review_analysis
                iter_found = iter_analysis.get("identified_count", 0)
                iter_accuracy = iter_analysis.get("identified_percentage", 0)
                
                feedback += f"| {iter_num} | {iter_found}/{total_problems} | {iter_accuracy:.1f}% |\n"
        
        # Tips for future reviews
        feedback += "\n## Tips for Future Code Reviews\n\n"
        
        # Derive tips from missed problems
        if missed_problems:
            problem_categories = set()
            
            # Simple categorization of missed problems
            for problem in missed_problems:
                problem_lower = problem.lower()
                
                if any(term in problem_lower for term in ["null", "nullpointer", "null pointer"]):
                    problem_categories.add("null pointer handling")
                elif any(term in problem_lower for term in ["array", "index", "bound"]):
                    problem_categories.add("array bounds checking")
                elif any(term in problem_lower for term in ["exception", "throw", "catch"]):
                    problem_categories.add("exception handling")
                elif any(term in problem_lower for term in ["comparison", "equals", "=="]):
                    problem_categories.add("object comparison")
                elif any(term in problem_lower for term in ["name", "convention", "style"]):
                    problem_categories.add("naming conventions")
                elif any(term in problem_lower for term in ["whitespace", "indent", "format"]):
                    problem_categories.add("code formatting")
                elif any(term in problem_lower for term in ["comment", "javadoc", "documentation"]):
                    problem_categories.add("code documentation")
                elif any(term in problem_lower for term in ["return", "type", "cast"]):
                    problem_categories.add("type safety")
                else:
                    problem_categories.add("general code quality")
            
            # Generate tips based on categories
            if problem_categories:
                feedback += "Based on your review, focus on these areas:\n\n"
                
                for category in problem_categories:
                    if category == "null pointer handling":
                        feedback += "- **Null Pointer Handling**: Always check if objects can be null before accessing their methods or properties.\n"
                    elif category == "array bounds checking":
                        feedback += "- **Array Bounds Checking**: Verify that array indices are within valid ranges before accessing elements.\n"
                    elif category == "exception handling":
                        feedback += "- **Exception Handling**: Look for code that might throw exceptions but doesn't handle them properly.\n"
                    elif category == "object comparison":
                        feedback += "- **Object Comparison**: Remember that `==` compares references while `equals()` compares content for objects like Strings.\n"
                    elif category == "naming conventions":
                        feedback += "- **Naming Conventions**: Verify that classes, methods, and variables follow Java naming conventions.\n"
                    elif category == "code formatting":
                        feedback += "- **Code Formatting**: Check for proper indentation, whitespace, and consistency in formatting.\n"
                    elif category == "code documentation":
                        feedback += "- **Documentation**: Ensure methods have proper Javadoc comments and parameters are documented.\n"
                    elif category == "type safety":
                        feedback += "- **Type Safety**: Look for improper type conversions or missing return statements.\n"
                    else:
                        feedback += "- **Code Quality**: Review for general coding best practices and potential bugs.\n"
        else:
            # Generic tips if no missed problems
            feedback += "- **Completeness**: Always check the entire codebase systematically.\n"
            feedback += "- **Methodology**: Develop a structured approach to code reviews.\n"
            feedback += "- **Documentation**: Look for clear and complete documentation.\n"
            feedback += "- **Best Practices**: Compare code against Java best practices.\n"
        
        # Add learning resources section
        if include_resources:
            feedback += "\n## Learning Resources\n\n"
            feedback += "To improve your Java code review skills, check out these resources:\n\n"
            
            # Add general resources
            feedback += "### General Code Review\n"
            feedback += "- [Code Review Best Practices](https://www.atlassian.com/agile/software-development/code-reviews)\n"
            feedback += "- [How to Do Code Reviews Like a Human](https://mtlynch.io/human-code-reviews-1/)\n"
            
            # Add Java-specific resources
            feedback += "\n### Java Specific\n"
            feedback += "- [Java Code Conventions](https://www.oracle.com/java/technologies/javase/codeconventions-introduction.html)\n"
            feedback += "- [Effective Java by Joshua Bloch](https://www.oreilly.com/library/view/effective-java-3rd/9780134686097/)\n"
            
            # Add tools and checklists
            feedback += "\n### Tools & Checklists\n"
            feedback += "- [PMD - Java static code analyzer](https://pmd.github.io/)\n"
            feedback += "- [CheckStyle - Code style enforcer](https://checkstyle.sourceforge.io/)\n"
            feedback += "- [Java Code Review Checklist](https://dev.to/javinpaul/my-favorite-free-courses-to-learn-java-in-depth-464k)\n"
        
        return feedback

    def reset(self):
        """Reset the feedback manager to initial state."""
        self.review_history = []
        self.current_iteration = 1
        self.review_sufficient = False
        self.code_snippet = ""
        self.known_problems = []