import re
import logging
import json
from typing import List, Dict, Any
from langchain_core.language_models import BaseLanguageModel

from utils.code_utils import format_list, create_review_analysis_prompt, create_feedback_prompt, process_llm_response
from utils.llm_logger import LLMInteractionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StudentResponseEvaluator:
    """
    Evaluates student code reviews against known problems in the code.
    
    This class analyzes how thoroughly and accurately a student identified 
    issues in a code snippet, providing detailed feedback and metrics.
    """    
    def __init__(self, llm: BaseLanguageModel = None,
                 min_identified_percentage: float = 60.0,
                 llm_logger: LLMInteractionLogger = None):
        """
        Initialize the StudentResponseEvaluator.
        
        Args:
            llm: Language model to use for evaluation
            min_identified_percentage: Minimum percentage of problems that
                                     should be identified for a sufficient review
            llm_logger: Logger for tracking LLM interactions
        """
        self.llm = llm
        self.min_identified_percentage = min_identified_percentage
        self.llm_logger = llm_logger or LLMInteractionLogger()
    
    def evaluate_review(self, code_snippet: str, known_problems: List[str], student_review: str, enhanced_errors: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a student's review with enhanced error detection and analysis.
        Uses the create_review_analysis_prompt function from code_utils.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            enhanced_errors: Optional enhanced error data with location information
            
        Returns:
            Dictionary with detailed analysis results
        """
        try:
            logger.info("Evaluating student review with code_utils prompt")
            
            if not self.llm:
                logger.warning("No LLM provided for evaluation, falling back to basic evaluation")
                return self._fallback_evaluation(known_problems)
            
            # Create a review analysis prompt using the utility function
            prompt = create_review_analysis_prompt(
                code=code_snippet,
                known_problems=known_problems,
                student_review=student_review
            )
            
            
            
            try:
                # Metadata for logging
                metadata = {
                    "code_length": len(code_snippet.splitlines()),
                    "known_problems_count": len(known_problems),
                    "student_review_length": len(student_review.splitlines()),
                    "has_enhanced_errors": enhanced_errors is not None
                }
                # Get the evaluation from the LLM
                logger.info("Sending student review to LLM for evaluation")
                response = self.llm.invoke(prompt)
                processed_response = process_llm_response(response)

                #self.llm_logger.log_student_comment_analysis(prompt, response, metadata)
                # Log the interaction
                self.llm_logger.log_review_analysis(prompt, processed_response, metadata)
                
                # Make sure we have a response
                if not response:
                    logger.error("LLM returned None or empty response for review evaluation")
                    return self._fallback_evaluation(known_problems)
                
                # Extract JSON data from the response
                analysis_data = self._extract_json_from_text(processed_response)
                
                # Make sure we have analysis data
                if not analysis_data or "error" in analysis_data:
                    logger.error(f"Failed to extract valid analysis data: {analysis_data.get('error', 'Unknown error')}")
                    return self._fallback_evaluation(known_problems)
                
                # Process the analysis data
                enhanced_analysis = self._process_enhanced_analysis(analysis_data, known_problems)
                
                # Add the original response for debugging
                enhanced_analysis["raw_llm_response"] = processed_response
                
                return enhanced_analysis
                
            except Exception as e:
                logger.error(f"Error evaluating review with LLM: {str(e)}")                
                # Log the error
                error_metadata = {**metadata, "error": str(e)}
                self.llm_logger.log_review_analysis(prompt, f"ERROR: {str(e)}", error_metadata)                
                return self._fallback_evaluation(known_problems)
            
        except Exception as e:
            logger.error(f"Exception in evaluate_review_enhanced: {str(e)}")
            return self._fallback_evaluation(known_problems)
            
    def _process_enhanced_analysis(self, analysis_data: Dict[str, Any], known_problems: List[str]) -> Dict[str, Any]:
        """
        Process and enhance the analysis data from the LLM.
        
        Args:
            analysis_data: Raw analysis data from LLM
            known_problems: List of known problems for reference
            
        Returns:
            Enhanced analysis data
        """
        # Handle None case
        if not analysis_data:
            return self._fallback_evaluation(known_problems)
        
        # Extract core metrics with defaults
        identified_count = analysis_data.get("identified_count", 0)
        missed_count = analysis_data.get("missed_count", 0)
        false_positive_count = analysis_data.get("false_positive_count", 0)
        total_problems = analysis_data.get("total_problems", len(known_problems))
        
        # Calculate percentages
        if total_problems > 0:
            identified_percentage = (identified_count / total_problems) * 100
        else:
            identified_percentage = 100.0
        
        # Extract review quality metrics
        review_quality_score = analysis_data.get("review_quality_score", 5.0)
        if not isinstance(review_quality_score, (int, float)):
            try:
                review_quality_score = float(review_quality_score)
            except:
                review_quality_score = 5.0
        
        # Determine if review is sufficient
        # Use LLM's determination if available, otherwise calculate based on percentage
        if "review_sufficient" in analysis_data:
            review_sufficient = analysis_data["review_sufficient"]
        else:
            review_sufficient = identified_percentage >= self.min_identified_percentage
        
        # Extract problem lists
        identified_problems = analysis_data.get("identified_problems", [])
        missed_problems = analysis_data.get("missed_problems", [])
        false_positives = analysis_data.get("false_positives", [])
        
        # Simplify the identified problems list if needed
        simple_identified = []
        for problem in identified_problems:
            if isinstance(problem, dict) and "problem" in problem:
                simple_identified.append(problem["problem"])
            elif isinstance(problem, str):
                simple_identified.append(problem)
        
        # Simplify the missed problems list if needed
        simple_missed = []
        for problem in missed_problems:
            if isinstance(problem, dict) and "problem" in problem:
                simple_missed.append(problem["problem"])
            elif isinstance(problem, str):
                simple_missed.append(problem)
        
        # Simplify the false positives list if needed
        simple_false_positives = []
        for false_positive in false_positives:
            if isinstance(false_positive, dict) and "student_comment" in false_positive:
                simple_false_positives.append(false_positive["student_comment"])
            elif isinstance(false_positive, str):
                simple_false_positives.append(false_positive)
        
        # Get overall feedback
        feedback = analysis_data.get("feedback", "")
        if not feedback:
            # Generate basic feedback based on performance
            if identified_percentage >= 80:
                feedback = "Excellent review! You found most of the issues in the code."
            elif identified_percentage >= 60:
                feedback = "Good review. You found many issues, but missed some important ones."
            elif identified_percentage >= 40:
                feedback = "Fair review. You found some issues, but missed many important ones."
            else:
                feedback = "Your review needs improvement. You missed most of the issues in the code."
        
        # Construct enhanced result
        enhanced_result = {
            "identified_problems": identified_problems,  # Keep the detailed version
            "missed_problems": missed_problems,  # Keep the detailed version
            "false_positives": false_positives,  # Keep the detailed version
            "simple_identified": simple_identified,  # Add simplified version
            "simple_missed": simple_missed,  # Add simplified version
            "simple_false_positives": simple_false_positives,  # Add simplified version
            "identified_count": identified_count,
            "missed_count": missed_count,
            "false_positive_count": false_positive_count,
            "total_problems": total_problems,
            "identified_percentage": identified_percentage,
            "accuracy_percentage": identified_percentage,  # For backward compatibility
            "review_quality_score": review_quality_score,
            "review_sufficient": review_sufficient,
            "feedback": feedback
        }
        
        return enhanced_result

    def _fallback_evaluation(self, known_problems: List[str]) -> Dict[str, Any]:
        """
        Generate a fallback evaluation when the LLM fails.
        
        Args:
            known_problems: List of known problems in the code
            
        Returns:
            Basic evaluation dictionary
        """
        logger.warning("Using fallback evaluation due to LLM error")
        
        # Create a basic fallback evaluation
        return {
            "identified_problems": [],
            "missed_problems": known_problems,
            "false_positives": [],
            "accuracy_percentage": 0.0,
            "identified_percentage": 0.0,
            "identified_count": 0,
            "total_problems": len(known_problems),
            "review_sufficient": False,
            "feedback": "Your review needs improvement. Try to identify more issues in the code."
        }
            
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON data from LLM response text.
        
        Args:
            text: Text containing JSON data
            
        Returns:
            Extracted JSON data
        """
        # Handle None or empty text
        if not text:
            return {"error": "Empty response from LLM"}
        
        try:
            # Try to find JSON block with regex
            patterns = [
                r'```json\s*([\s\S]*?)```',  # JSON code block
                r'```\s*({[\s\S]*?})\s*```',  # Any JSON object in code block
                r'({[\s\S]*"identified_problems"[\s\S]*"missed_problems"[\s\S]*})',  # Look for our expected fields
                r'({[\s\S]*})',  # Any JSON-like structure
            ]
            
            # Try each pattern
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean up the match
                        json_str = match.strip()
                        # Try to parse as JSON
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # If standard methods fail, try to manually extract fields
            logger.warning("Could not extract JSON, attempting manual extraction")
            analysis = {}
            
            # Try to extract identified problems
            identified_match = re.search(r'"identified_problems"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if identified_match:
                try:
                    identified_str = identified_match.group(1)
                    analysis["identified_problems"] = json.loads(identified_str)
                except:
                    analysis["identified_problems"] = []
            else:
                analysis["identified_problems"] = []
            
            # Try to extract missed problems
            missed_match = re.search(r'"missed_problems"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if missed_match:
                try:
                    missed_str = missed_match.group(1)
                    analysis["missed_problems"] = json.loads(missed_str)
                except:
                    analysis["missed_problems"] = []
            else:
                analysis["missed_problems"] = []
            
            # Try to extract false positives
            false_pos_match = re.search(r'"false_positives"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if false_pos_match:
                try:
                    false_pos_str = false_pos_match.group(1)
                    analysis["false_positives"] = json.loads(false_pos_str)
                except:
                    analysis["false_positives"] = []
            else:
                analysis["false_positives"] = []
            
            # Try to extract accuracy percentage
            accuracy_match = re.search(r'"accuracy_percentage"\s*:\s*([0-9.]+)', text)
            if accuracy_match:
                try:
                    analysis["accuracy_percentage"] = float(accuracy_match.group(1))
                except:
                    analysis["accuracy_percentage"] = 0.0
            else:
                analysis["accuracy_percentage"] = 0.0
            
            # Try to extract review_sufficient
            sufficient_match = re.search(r'"review_sufficient"\s*:\s*(true|false)', text)
            if sufficient_match:
                analysis["review_sufficient"] = sufficient_match.group(1) == "true"
            else:
                analysis["review_sufficient"] = False
            
            # Try to extract feedback
            feedback_match = re.search(r'"feedback"\s*:\s*"(.*?)"', text)
            if feedback_match:
                analysis["feedback"] = feedback_match.group(1)
            else:
                analysis["feedback"] = "The analysis could not extract feedback."
            
            if analysis:
                return analysis
            
            # If all else fails, return an error object
            logger.error("Could not extract analysis data from LLM response")
            return {
                "error": "Could not parse JSON response",
                "raw_text": text[:500] + ("..." if len(text) > 500 else "")
            }
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return {
                "error": f"Error extracting JSON: {str(e)}",
                "raw_text": text[:500] + ("..." if len(text) > 500 else "")
            }
       
    def generate_targeted_guidance_enhanced(self, code_snippet: str, known_problems: List[str], student_review: str, review_analysis: Dict[str, Any], iteration_count: int, max_iterations: int, enhanced_errors: List[Dict[str, Any]] = None) -> str:
        """
        Generate enhanced targeted guidance for the student to improve their review.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            review_analysis: Analysis of the student review
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            enhanced_errors: Optional enhanced error information            
        Returns:
            Targeted guidance text
        """        
        if not self.llm:
            logger.warning("No LLM provided for guidance generation, falling back to simple guidance")
            return "Try to identify more issues in the code. Look for naming convention violations, logical errors, and potential runtime exceptions."
        
          
        try:
            # Get iteration information to add to review_analysis for context
            review_context = review_analysis.copy()
            review_context.update({
                "iteration_count": iteration_count,
                "max_iterations": max_iterations,
                "remaining_attempts": max_iterations - iteration_count
            })

             # Use the utility function to create the prompt
            prompt = create_feedback_prompt(
                code=code_snippet,
                known_problems=known_problems,
                review_analysis=review_context
            )

            metadata = {
                "iteration": iteration_count,
                "max_iterations": max_iterations,
                "identified_count": review_analysis.get("identified_count", 0),
                "total_problems": review_analysis.get("total_problems", len(known_problems)),
                "identified_percentage": review_analysis.get("identified_percentage", 0)
            }

            # Generate the guidance using the LLM
            logger.info(f"Generating targeted guidance using code_utils prompt for iteration {iteration_count}")
            response = self.llm.invoke(prompt)
            guidance = process_llm_response(response)
            
            # Log the interaction
            self.llm_logger.log_summary_generation(prompt, guidance, metadata)            
            return guidance
            
        except Exception as e:
            logger.error(f"Error generating enhanced guidance with LLM: {str(e)}")            
            # Log the error
            error_metadata = {**metadata, "error": str(e)}
            self.llm_logger.log_interaction("targeted_guidance", prompt, f"ERROR: {str(e)}", error_metadata)            
            # Fallback to simple guidance
            return f"You've identified {review_analysis.get('identified_count', 0)} out of {review_analysis.get('total_problems', len(known_problems))} issues. Try to look more carefully at the code and identify additional issues. Pay attention to naming conventions, potential null references, and logical errors."