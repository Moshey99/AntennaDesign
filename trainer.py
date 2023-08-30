import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np


def train_model_single_epoch(model, loss_fn, optimizer, train_loader, epoch, grad_accumulation_step = None, clip_norm = 1):
    model.train()

    running_loss = 0
    num_batches_per_print = 100
    for idx, sample in enumerate(train_loader):
        data, target = (sample[0], sample[1]), sample[2]
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)  # average loss over batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm )
        if grad_accumulation_step is not None:
            if (idx + 1) % grad_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        if (idx+1) % num_batches_per_print == 0:
            avg_loss = running_loss / num_batches_per_print
            print('TRAIN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (idx+1) * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * (idx+1) * train_loader.batch_size / len(
                                                                               train_loader.dataset), avg_loss))
            running_loss = 0
    if grad_accumulation_step > 1 and (idx + 1) % grad_accumulation_step != 0:
        optimizer.step()
    return avg_loss


def evaluate_model(model, loss_fn, data_loader, set):
    model.eval()
    all_losses = np.array([])
    with torch.no_grad():
        for idx, sample in enumerate(data_loader):
            data, target = (sample[0], sample[1]), sample[2]
            output = model(data)
            loss = loss_fn(output, target).item()
            all_losses = np.append(all_losses, loss)
    avg_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    print('\n{} set: Average loss: {:.4f} +- {:.4f}'.format(set, avg_loss, std_loss))
    return avg_loss, std_loss


def run_model(model, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, schedule_step, gamma,grad_accumulation_step = None):
    train_losses, train_losses_std, val_losses, val_losses_std = [], [], [], []
    # before training:
    val_loss, val_loss_std = evaluate_model(model, loss_fn, val_loader, set='Validation')
    train_loss, train_loss_std = evaluate_model(model, loss_fn, train_loader, set='Train')
    train_losses.append(train_loss)
    train_losses_std.append(train_loss_std)
    val_losses.append(val_loss)
    val_losses_std.append(val_loss_std)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=schedule_step, gamma=gamma)
    print('Starting training')
    for epoch in range(1, epochs + 1):
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        _ = train_model_single_epoch(model, loss_fn, optimizer, train_loader, epoch, grad_accumulation_step)
        scheduler.step()
        val_loss, val_loss_std = evaluate_model(model, loss_fn, val_loader, set='Validation')
        train_loss, train_loss_std = evaluate_model(model, loss_fn, train_loader, set='Train')

        train_losses.append(train_loss)
        train_losses_std.append(train_loss_std)
        val_losses.append(val_loss)
        val_losses_std.append(val_loss_std)
    test_loss, _ = evaluate_model(model, loss_fn, test_loader, set='Test')
    return train_losses, train_losses_std, val_losses, val_losses_std, test_loss
