from lightrag_ollama_demo import LightRagManager

if __name__ == '__main__':
    db = "./lr_db"
    manager = LightRagManager(db)
    manager.visualize()