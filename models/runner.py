from tqdm.auto import tqdm as tq
import numpy as np
import torch

class Runner():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def train(self, train_loader, valid_loader,
              optimizer, scheduler,
              valid_score_fn,
              n_epochs, train_on_gpu=False, verbose=False, save_rst=True):

        if train_on_gpu:
            self.model.cuda()

        train_loss_list, valid_loss_list, dice_score_list = [], [], []
        lr_rate_list = []
        valid_loss_min = np.Inf
        for epoch in range(1, n_epochs + 1):
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            dice_score = 0.0

            ###################
            # train the model #
            ###################
            self.model.train()

            bar = tq(train_loader, postfix={"train_loss": 0.0})
            for data, target in bar:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)

                # calculate the batch loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()

                train_loss += loss.item() * data.size(0)
                # print("Loss item: {}, data_size:{}".format(loss.item(), data.size(0)))
                bar.set_postfix(ordered_dict={"train_loss": loss.item()})

            ######################
            # validate the model #
            ######################
            self.model.eval()
            del data, target
            with torch.no_grad():
                bar = tq(valid_loader, postfix={"valid_loss": 0.0, "dice_score": 0.0})
                for data, target in bar:
                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    loss = self.criterion(output, target)
                    # update average validation loss
                    valid_loss += loss.item() * data.size(0)
                    dice_cof = valid_score_fn(output.cpu(), target.cpu()).item()
                    dice_score += dice_cof * data.size(0)
                    bar.set_postfix(ordered_dict={"valid_loss": loss.item(), "dice_score": dice_cof})

            # calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            dice_score = dice_score / len(valid_loader.dataset)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            dice_score_list.append(dice_score)
            lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])

            # print training/validation statistics
            print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f}'.format(
                epoch, train_loss, valid_loss, dice_score))

            if save_rst:
                with open('training_rst.txt', 'w') as frst:
                    frst.write(str(train_loss_list) + '\n')
                    frst.write(str(valid_loss_list) + '\n')
                    frst.write(str(dice_score_list) + '\n')

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                    torch.save(self.model.state_dict(), 'model_cifar.pt')
                    valid_loss_min = valid_loss

            scheduler.step(valid_loss)

        return train_loss_list, valid_loss_list, dice_score_list, lr_rate_list