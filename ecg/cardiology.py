import numpy as np
from collections import Counter

y = np.load("/Users/omarelsisi/Downloads/medical_ai/y.npy", allow_pickle=True)

print(type(y))
print(y.shape)
print(y[:20])
print(np.unique(y))

counts = Counter(y)
print(counts)