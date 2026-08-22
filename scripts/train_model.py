from src.model.train import train

if __name__ == "__main__":
    model, history, class_names = train()
    print(f"Training completed. Classes: {class_names}")
