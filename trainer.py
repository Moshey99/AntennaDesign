import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
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
        if batch_idx % 100 == 0:
            print('TRAIN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader[0]),
                                                                           100. * batch_idx * len(data) / len(train_loader[0]),
                                                                           loss.item()))
    return loss.item()
def evaluate_model(model,loss_fn,data_loader,device,set):
    model.eval()
    all_losses = np.array([])
    with torch.no_grad():
        for idx in range(len(data_loader[0])):
            data,target = data_loader[0][idx].to(device).float(),data_loader[1][idx].to(device).float()
            output = model(data)
            loss = loss_fn(output,target).item()
            all_losses = np.append(all_losses,loss)
    avg_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    print('\n{} set: Average loss: {:.4f} +- {:.4f}'.format(set,avg_loss,std_loss))
    return avg_loss,std_loss





def run_model(model,loss_fn,optimizer,train_loader,val_loader,test_loader,epochs,batch_size,schedule_step,device):
    train_losses,val_losses = [],[]
    scheduler = lr_scheduler.StepLR(optimizer, step_size=schedule_step, gamma=0.2)
    print('Starting training')
    for epoch in range(1,epochs+1):
        if epoch>10:
            scheduler.step()
            print('Epoch:',epoch,'LR:',scheduler.get_lr())
        train_loss = train_model_single_epoch(model,loss_fn,optimizer,train_loader,epoch,batch_size,device)
        val_loss,val_loss_std = evaluate_model(model,loss_fn,val_loader,device,set='Validation')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    evaluate_model(model,loss_fn,test_loader,device,set='Test')
    return train_losses,val_losses