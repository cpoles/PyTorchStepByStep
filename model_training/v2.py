
# Defines number of epochs
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    # store loss for each mini-batch
    mini_batch_losses = []
    # iterate through the train_loader
    for x_batch, y_batch in train_loader:
        # the dataset is in the CPU, so are the mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Performs one train step and returns the
        # corresponding loss for this mini-batch
        mini_batch_loss = train_step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
        
    # Computes the average loss over all mini-batches
    # That's the epoch loss
    loss = np.mean(mini_batch_losses)
    
    losses.append(loss)    
