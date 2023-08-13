
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sets learning rate
lr = 0.1

torch.manual_seed(42)

# Create model and send to device
model = nn.Sequential(nn.Linear(1,1)).to(device)

# Defines an SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines an MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Creates the train step function using model , loss_fn, and optimizedr
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# Creates the val step function 
val_step_fn = make_val_step_fn(model, loss_fn)

# Creates a SummaryWriter to interface with TensorBoard
writer = SummaryWriter('runs/simple_linear_regression')

# fetches a single mini-batch so we can use add_graph
x_dummy,  y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))
