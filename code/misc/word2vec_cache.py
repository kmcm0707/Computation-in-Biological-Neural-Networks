import os

import gensim.downloader as api
import torch


def create_word2vec_cache(save_path="data/word2vec_300.pt"):
    save_path = os.getcwd() + "/" + save_path

    if not os.path.exists(os.getcwd() + "/data"):
        os.makedirs(os.getcwd() + "/data")

    print("Downloading Google News Word2Vec (1.6GB)...")
    wv = api.load("word2vec-google-news-300")

    # We add one extra row at the end for Unknown words (index -1)
    weights = torch.FloatTensor(wv.vectors)
    unk_vector = torch.mean(weights, dim=0, keepdim=True)  # Average vector for UNK
    weights = torch.cat([weights, unk_vector], dim=0)

    # vocab mapping: word -> index
    vocab = {word: index for index, word in enumerate(wv.index_to_key)}

    print(f"Saving cache. Total vocab size: {len(weights)}")
    torch.save({"weights": weights, "vocab": vocab}, save_path)


if __name__ == "__main__":
    create_word2vec_cache()
