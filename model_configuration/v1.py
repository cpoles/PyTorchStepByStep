
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sets learning rate
lr = 0.1

torch.manual_seed(42)
# Now we can create a model and send it at once to the device
model = nn.Sequential(
    nn.Linear(1, 1)
).to(device)

# Defines an SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines an MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Creates the train_step function for our model, loss function and optimizer
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
