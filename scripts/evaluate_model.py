import json
from src.model.evaluate import evaluate

if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({
        "test_loss": result["test_loss"],
        "test_accuracy": result["test_accuracy"],
    }, indent=2))
