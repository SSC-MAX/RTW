import numpy as np
import json
import torch
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class SentenceEmbeddings:

    def __init__(self, model_path):
        print(f'model_path=>{model_path}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        
        
    def get_embedding(self, sentence):
        """generate embedding of single sentence"""
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
        return output[0][:, 0, :].cpu().numpy()

    def generate_embeddings(self, input_path, output_path, generate_size=1000):
        """generate embedding for each sentence in the input text"""
        all_embeddings = []
        with open(input_path, 'r') as f:
            lines = f.readlines()

        pbar = tqdm(total=generate_size, desc="Embeddings generated")

        for line in lines:
            data = json.loads(line)
            all_embeddings.append(self.get_embedding(data['sentence1']))
            all_embeddings.append(self.get_embedding(data['sentence2']))
            pbar.update(2)
            if len(all_embeddings) >= generate_size:
                break

        pbar.close()

        all_embeddings = np.vstack(all_embeddings)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, all_embeddings, delimiter=" ")


if __name__ == '__main__':
    input_path = '../train_data/train.jsonl'
    output_path = '../train_data/train_embeddings.txt'
    model_path = 'google-bert/bert-large-uncased'
    size = 1000

    sentence_embeddings = SentenceEmbeddings(model_path)
    sentence_embeddings.generate_embeddings(input_path, output_path, size)

    print('===Completed===')
