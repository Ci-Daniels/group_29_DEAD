import numpy as np

embedding = np.load(
    "./data/embeddings/WEB001/sample_00.npy"
)

print(embedding)
print(type(embedding))
print(embedding.shape)
print(embedding.dtype)