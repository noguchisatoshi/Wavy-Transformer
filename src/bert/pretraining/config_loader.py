# config_loader.py
import yaml

def load_config(config_path="bert/pretraining/pretraining_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)