
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sets learning rate
lr = 0.1

torch.manual_seed(42)
# Create model and send it to device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Defines SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines MSE loss function
loss_fn = nn.MSELoss(reduction='mean')
