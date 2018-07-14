from MSRADataset import MSRADataset
from MSRADataset import read_depth_from_bin, get_center, _crop_image, normalize, read_joints
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
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.0005 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def main(resume=False):
    model = REN()
    model.double()
    model.cuda()
    cudnn.benchmark = True
    criterion = nn.SmoothL1Loss().cuda()

    train_dataset = MSRADataset()
    test_dataset = MSRADataset(False)
    optimizer = torch.optim.SGD(model.parameters(), 0.0005,
                                momentum=0.9,
                                weight_decay=0.0005)





    if resume:
        val_loader = torch.utils.data.DataLoader(
           test_dataset, batch_size= 64 ,
           num_workers=4, pin_memory=True)

        model, optimizer = load_checkpoint("1_checkpoint.pth.tar", model, optimizer)
        validate(val_loader, model, criterion)
        return

    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=64, shuffle = True,
       num_workers=4, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=64  ,shuffle = True,
       num_workers=4, pin_memory=True)

    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in range(0,150):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch)
        train_loss = train_loss + loss_train
        # evaluate on validation set
        loss_val = validate(val_loader, model, criterion)
        val_loss = val_loss + loss_val

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

            target = target.double()
            target = target.cuda(non_blocking=True)
            # compute output
            input = input.double()
            input = input.cuda()
            output = model(input)
            loss = criterion(output, target)

            if i % 5 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                       i, len(val_loader), loss=loss))
            loss_val.append(loss.data[0])


    return [np.mean(loss_val)]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(path,model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


if __name__ == '__main__':
    main()
    # torch.no_grad()
    # # torch.set_printoptions(threshold=100000)
    # np.set_printoptions(threshold=np.nan)
    # depth = read_depth_from_bin("data/P0/5/000000_depth.bin")
    # #get centers
    # center = get_center(depth)
    # #get cube and resize to 96x96
    # depth = _crop_image(depth, center, is_debug=False)
    # # print(depth)
    # assert ((depth>1).sum() == 0)
    # assert ((depth<-1).sum() == 0)
    # #normalize
    # # depth1 = normalize(depth)
    # # print(np.array_equal(depth,depth1))
    # depth = (torch.from_numpy(depth))
    # depth = torch.unsqueeze(depth, 0)
    # depth = torch.unsqueeze(depth, 0)
    # # print(depth.shape)
    #
    # depth1 = read_depth_from_bin("data/P2/5/000400_depth.bin")
    # #get centers
    # center = get_center(depth1)
    # #get cube and resize to 96x96
    # depth1 = _crop_image(depth1, center, is_debug=False)
    # # print(depth)
    # assert ((depth1>1).sum() == 0)
    # assert ((depth1<-1).sum() == 0)
    # # normalize
    # # depth1 = normalize(depth)
    # print(np.array_equal(depth,depth1))
    # depth1 = (torch.from_numpy(depth1))
    # depth1 = torch.unsqueeze(depth1, 0)
    # depth1= torch.unsqueeze(depth1, 0)
    #
    # print(torch.eq(depth1,depth).all())
    # model = REN()
    # criterion = nn.SmoothL1Loss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), 0.005,
    #                             momentum=0.9,
    #                             weight_decay=0.0005)
    #
    # model, optimizer = load_checkpoint("checkpoint.pth.tar", model, optimizer)
    # model.eval()
    # model = model.train(False)
    # model =model.cuda()
    # model =model.double()
    # depth = depth.cuda()
    # depth = depth.double()
    # depth1 = depth1.cuda()
    # depth1 = depth1.double()
    #
    # test = np.ones((96,96))
    # test = (torch.from_numpy(test))
    # test = torch.unsqueeze(test, 0)
    # test= torch.unsqueeze(test, 0)
    # test = test.cuda()
    # test = test.double()
    #
    #
    # # print(depth)
    # results = model(depth)
    # results1 = model(depth1)
    # results2 = model(test)
    # joints = read_joints()
    # print(joints[0])
    # print(results)
    # # print(results1)
    # # print(results2)
    # # print(torch.eq(results,results1).all())
    # # print(torch.eq(results2,results1).all())
    # # print(list(model.parameters()))
