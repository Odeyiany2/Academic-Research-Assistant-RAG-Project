import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging 

# Set up logging for evaluation
eval_log_filename = "evals.log"
eval_logger = logging.getLogger("eval_logger")
eval_logger.setLevel(logging.INFO)
eval_handler = logging.FileHandler(eval_log_filename)
eval_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
eval_logger.addHandler(eval_handler)


test_queries = [
    {"query": "What is contract law and it's importance in business?", "expected_answer": "Contract law is the body of law governing the creation and enforcement of contracts."},
    {"query": "What is contract law and it's importance in business?", "expected_answer": "Contract law is at the centre of most human activities, governing the relationships and agreements formed between individuals and businesses. It provides a framework for understanding and navigating contractual relationships, ensuring that parties fulfill their obligations and responsibilities. The law of contract is essential in business as it provides a basis for trade and commerce, allowing companies to engage in transactions and partnerships with confidence. By establishing clear rules and guidelines, contract law promotes trust, stability, and predictability in business dealings. Ultimately, contract law plays a vital role in facilitating economic growth and development by enabling businesses to operate effectively and efficiently."}
]


# Load pre-trained model and tokenizer for embeddings (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to get sentence embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Average pooling to get single vector

# Function to calculate cosine similarity between two sentences
def calculate_cosine_similarity(response_1, response_2):
    embedding_1 = get_embedding(response_1)
    embedding_2 = get_embedding(response_2)
    # Calculate cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embedding_1, embedding_2)
    return cosine_sim.item()



def evaluate_model_response(query, model_name, result):
    total_cosine_similarity = 0
    total_queries = len(test_queries)

    for item in test_queries:
        query = item["query"]
        expected_answer = item["expected_answer"]
        try:
            model_response = result
            
            # Calculate cosine similarity between model response and expected answer
            cosine_sim = calculate_cosine_similarity(model_response, expected_answer)
            total_cosine_similarity += cosine_sim
        except Exception as e:
            eval_logger.error(f"Error while evaluating query '{query}': {e}")

    # Calculate average cosine similarity score
    average_cosine_similarity = total_cosine_similarity / total_queries
    eval_logger.info(f"Average Cosine Similarity of {model_name} is: {average_cosine_similarity}")

