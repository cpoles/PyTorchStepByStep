
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sets learning rate
lr = 0.1

torch.manual_seed(42)
# Send model to device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Defines SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines an MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Creates the train_step function using the model, optimizer and loss_fn
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# Create the val_step function using the model and loss function
val_step_fn = make_val_step_fn(model, loss_fn)
