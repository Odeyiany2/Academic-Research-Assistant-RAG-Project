from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from src.exceptions.operationshandler import *


def evaluate_model():
    #actual_output = 

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="llama-3.1-70b-versatile",
        include_reason=True
    )

    test_case = LLMTestCase(
        #input=query,
        #actual_output=actual_output
    )

    evaluation_logger.log(metric.measure(test_case))
    #print(metric.score)
    #print(metric.reason)
