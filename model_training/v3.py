
# defines number of epochs
n_epochs = 200

losses = []

for epoch in range(n_epochs):
    # mini batch gradient descent
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)
