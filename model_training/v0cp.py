
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Sets model to train mode
    model.train()
    
    # Step 1 - Computes model's predicted output - forward pass
    yhat = model(x_train_tensor)
    
    # Step 2 - Computes loss
    loss = loss_fn(yhat, y_train_tensor)
    
    # Step 3 - Computes gradients
    loss.backward()
    
    # Step 4 - Updates parameters
    optimizer.step()
    optimizer.zero_grad()
