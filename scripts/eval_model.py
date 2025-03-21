from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Define topic vocabulary as a constant
TOPIC_VOCAB = ["price", "quality", "design", "function", "customer service", "other"]

def evaluate_model_multi_label(true_data, pred_data):
    """
    Evaluate model performance for multi-label topic classification with per-topic sentiment scores.
    
    Args:
        true_data (List[Dict[str, float]]): Ground-truth data. Each dict maps topics to sentiment scores.
            Example: [{"politics": 0.2, "technology": 0.8}, {...}]
        pred_data (List[Dict[str, float]]): Predicted data in same format as true_data
        
    Returns:
        dict: Evaluation metrics
    """
    all_true_topics = []
    all_pred_topics = []
    sentiment_errors = []
    correctly_predicted_topics = 0
    total_true_topics = 0
    
    for true_item, pred_item in zip(true_data, pred_data):
        true_topics = list(true_item.keys())
        pred_topics = list(pred_item.keys())
        
        true_binary = [1 if t in true_topics else 0 for t in TOPIC_VOCAB]
        pred_binary = [1 if t in pred_topics else 0 for t in TOPIC_VOCAB]
        
        all_true_topics.append(true_binary)
        all_pred_topics.append(pred_binary)
        
        total_true_topics += len(true_topics)
        
        for topic in true_topics:
            if topic in pred_topics:
                correctly_predicted_topics += 1
                true_sentiment = true_item[topic]
                pred_sentiment = pred_item[topic]
                sentiment_errors.append((true_sentiment - pred_sentiment) ** 2)
    
    # Convert to numpy arrays
    all_true_topics = np.array(all_true_topics)
    all_pred_topics = np.array(all_pred_topics)
    
    results = {
        'Topic_Precision': precision_score(all_true_topics, all_pred_topics, average='samples', zero_division=0),
        'Topic_Recall': recall_score(all_true_topics, all_pred_topics, average='samples', zero_division=0),
        'Topic_F1': f1_score(all_true_topics, all_pred_topics, average='samples', zero_division=0),
        'Sentiment_MSE': np.mean(sentiment_errors) if sentiment_errors else float('nan'),
        'Topic_Sentiment_Coverage': correctly_predicted_topics / total_true_topics if total_true_topics > 0 else 0.0
    }
    
    return results

if __name__ == "__main__":
    # Sample ground truth data with direct mapping of topics to sentiment scores
    true_data = [
        {"politics": 0.2, "technology": 0.8},
        {"sports": 0.9, "health": 0.6},
        {"entertainment": 0.7, "politics": -0.3}
    ]
    
    # Sample predicted data with direct mapping of topics to sentiment scores
    pred_data = [
        {"politics": 0.3, "entertainment": 0.5},
        {"sports": 0.8, "technology": 0.2},
        {"entertainment": 0.6, "politics": -0.4, "health": 0.1}
    ]
    
    # Evaluate the model
    results = evaluate_model_multi_label(true_data, pred_data)
    
    # Print the results
    print("Multi-label Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
