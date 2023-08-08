
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data originally in numpy arrays. We need to transform them into PyTorch's tensors and send it to the gpu
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
