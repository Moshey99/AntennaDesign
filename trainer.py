import torch
def train_model_single_epoch(model,loss_fn,optimizer,train_loader,epoch,batch_size,device):
    model.train()
    num_batches = len(train_loader[0])//batch_size
    for batch_idx in range(num_batches):
        data, target = train_loader[0][batch_idx*batch_size:(batch_idx+1)*batch_size],train_loader[1][batch_idx*batch_size:(batch_idx+1)*batch_size]
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader[0]),
                                                                           100. * batch_idx * len(data) / len(train_loader[0]),
                                                                           loss.item()))

def evaluate_model(model,loss_fn,test_loader,batch_size,device):
    model.eval()
    test_loss = 0
    num_batches = len(test_loader[0])//batch_size
    with torch.no_grad():
        for batch_idx in range(num_batches):
            data, target = test_loader[0][batch_idx*batch_size:(batch_idx+1)*batch_size],test_loader[1][batch_idx*batch_size:(batch_idx+1)*batch_size]
            data,target = data.to(device).float(),target.to(device).float()
            output = model(data)
            test_loss += loss_fn(output,target).item()
    test_loss /= num_batches
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

def run_model(model,loss_fn,optimizer,train_loader,val_loader,test_loader,epochs,batch_size,device):

    print('Starting training')
    for epoch in range(1,epochs+1):
        train_model_single_epoch(model,loss_fn,optimizer,train_loader,epoch,batch_size,device)
        evaluate_model(model,loss_fn,val_loader,batch_size,device)
    evaluate_model(model,loss_fn,test_loader,batch_size,device)