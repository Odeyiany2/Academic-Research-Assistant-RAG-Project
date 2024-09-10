from src.exceptions.operationshandler import *
import requests
# Assuming you have a list of queries and their expected answers
test_queries = [
    {"query": "What is contract law and it's importance in business?", "expected_answer": "Contract law is the body of law governing the creation and enforcement of contracts."},
    {"query": "What is contract law and it's importance in business?", "expected_answer": "Contract law is at the centre of most human activities, governing the relationships and agreements formed between individuals and businesses. It provides a framework for understanding and navigating contractual relationships, ensuring that parties fulfill their obligations and responsibilities. The law of contract is essential in business as it provides a basis for trade and commerce, allowing companies to engage in transactions and partnerships with confidence. By establishing clear rules and guidelines, contract law promotes trust, stability, and predictability in business dealings. Ultimately, contract law plays a vital role in facilitating economic growth and development by enabling businesses to operate effectively and efficiently."}
]

# Function to get model's response (this is a placeholder, replace with your actual function)
def get_model_response(query, model_name):
    response = requests.post("http://127.0.0.1:5000/query", json={"question": query, "model": model_name})
    if response.status_code == 200:
        return response.json().get("result", "")
    else:
        raise Exception(f"Model request failed with status code {response.status_code}")
def evaluate_model_response(query, model_name):
    correct_answers = 0
    total_queries = len(test_queries)

    for item in test_queries:
        query = item["query"]
        expected_answer = item["expected_answer"]
        try:
            model_response = get_model_response(query, model_name)
            
            if model_response in expected_answer:
                correct_answers += 1
        except Exception as e:
            evaluation_logger.log(f"Error while evaluating query '{query}': {e}")

    # Calculate accuracy
    accuracy = (correct_answers / total_queries) * 100
    evaluation_logger.log(f"Accuracy of the {model_name} is: {accuracy}")
