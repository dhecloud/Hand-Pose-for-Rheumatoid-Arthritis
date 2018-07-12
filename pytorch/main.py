from MSRADataset import MSRADataset
from REN import REN
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time


OUTFILE = "results"
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.005 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def main():
    model = REN()
    model.double()
    model.cuda()
    print(model)
    print(next(model.parameters()).is_cuda)
    cudnn.benchmark = True
    criterion = nn.SmoothL1Loss().cuda()

    train_dataset = MSRADataset()
    test_dataset = MSRADataset(False)
    optimizer = torch.optim.SGD(model.parameters(), 0.005,
                                momentum=0.9,
                                weight_decay=0.0005)

    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=64,
       num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=64  ,
       num_workers=4, pin_memory=True)

    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in range(0, 1000):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch)
        train_loss = train_loss + loss_train
        # evaluate on validation set
        loss_val = validate(val_loader, model, criterion)
        val_loss = val_loss + loss_val
        # remember best prec@1 and save checkpoint

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "REN",
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, False)

    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_train_loss.out",train_loss)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "val_loss.out",val_loss)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "val_acc.out",val_acc)



def train(train_loader, model, criterion, optimizer, epoch):

    # switch to train mode
    model.train()
    loss_train = []
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        target = target.double()
        target = target.cuda(non_blocking=True)
        input = input.double()
        input = input.cuda()
        # compute output
        output = model(input)

        loss = criterion(output, target)
        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        loss_train.append(loss.data[0])
        optimizer.step()

        # measure elapsed time
        if i % 30 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                   epoch, i, len(train_loader), loss=loss.data[0]))

    return [np.mean(loss_train)]



def validate(val_loader, model, criterion):

    # switch to evaluate mode
    model.eval()

    loss_val = []

    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):

            target=target.double()
            target = target.cuda(non_blocking=True)
            # compute output
            input = input.double()
            input = input.cuda()
            output = model(input)

            loss = criterion(output, target)

            if i % 30 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                       i, len(val_loader), loss=loss))
            loss_val.append(loss.data[0])


    return [np.mean(loss_val)]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()
