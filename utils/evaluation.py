#from src.exceptions.operationshandler import evaluation_logger
import requests
import logging 

# Set up logging for evaluation
eval_log_filename = "evals.log"
eval_logger = logging.getLogger("eval_logger")
eval_logger.setLevel(logging.INFO)
eval_handler = logging.FileHandler(eval_log_filename)
eval_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
eval_logger.addHandler(eval_handler)



# Assuming you have a list of queries and their expected answers
test_queries = [
    {"query": "What is contract law and it's importance in business?", "expected_answer": "Contract law is the body of law governing the creation and enforcement of contracts."},
    {"query": "What is contract law and it's importance in business?", "expected_answer": "Contract law is at the centre of most human activities, governing the relationships and agreements formed between individuals and businesses. It provides a framework for understanding and navigating contractual relationships, ensuring that parties fulfill their obligations and responsibilities. The law of contract is essential in business as it provides a basis for trade and commerce, allowing companies to engage in transactions and partnerships with confidence. By establishing clear rules and guidelines, contract law promotes trust, stability, and predictability in business dealings. Ultimately, contract law plays a vital role in facilitating economic growth and development by enabling businesses to operate effectively and efficiently."}
]

# Function to get model's response (this is a placeholder, replace with your actual function)
# def get_model_response(query, model_name):
#     response = requests.post("http://127.0.0.1:5000/query", json={"question": query, "model": model_name})
#     if response.status_code == 200:
#         return response.json().get("result", "")
#     else:
#         raise Exception(f"Model request failed with status code {response.status_code}")
def evaluate_model_response(query, model_name, result):
    correct_answers = 0
    total_queries = len(test_queries)

    for item in test_queries:
        query = item["query"]
        expected_answer = item["expected_answer"]
        try:
            model_response = result
            
            if model_response in expected_answer:
                correct_answers += 1
        except Exception as e:
            eval_logger.error(f"Error while evaluating query '{query}': {e}")

    # Calculate accuracy
    accuracy = (correct_answers / total_queries) * 100
    eval_logger.info(f"Accuracy of the {model_name} is: {accuracy}")

#evaluate_model_response(model_name="llama-3.1-70b-versatile", )
