import torch as tr


DEVICE = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

print(DEVICE)