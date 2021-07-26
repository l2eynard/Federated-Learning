from numpy import load

data = load('round-1-weights.npz', allow_pickle=True)
lst = data.files
for item in lst:
    print(item)
    print(data[item])