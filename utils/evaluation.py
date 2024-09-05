from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from src.exceptions.operationshandler import *
from utils.helpers import *
from app import query_model


def evaluate_model(query, model_response):
    try:
        actual_output = model_response

        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model="llama-3.1-70b-versatile",
            include_reason=True
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output
        )

        # Evaluate and get results
        evaluation_result = metric.measure(test_case)

        # Log the evaluation results
        evaluation_logger.info(f"Evaluation for query: {query}")
        evaluation_logger.info(f"Score: {evaluation_result.score}")
        evaluation_logger.info(f"Reason: {evaluation_result.reason}")

    except Exception as e:
        evaluation_logger.error(f"Error during evaluation: {str(e)}")
        raise QueryEngineError("Failed to evaluate model.") from e
