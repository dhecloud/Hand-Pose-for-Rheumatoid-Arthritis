from MSRADataset import MSRADataset
from MSRADataset import read_depth_from_bin, get_center, _crop_image, read_joints, augment_translation, augment_scaling, augment_rotation
from REN import REN
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time
import cv2

print_interval = 100
OUTFILE = "results"

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    # if epoch < 11:
    #     lr = 0.0005 * (0.1 ** (epoch // 10))
    # else:
    lr = 0.0005 * (0.1 ** (epoch // 1))
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
           num_workers=2, pin_memory=True)

        model, optimizer = load_checkpoint("1_checkpoint.pth.tar", model, optimizer)
        validate(val_loader, model, criterion)
        return

    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=64, shuffle = True,
       num_workers=2, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=64  ,shuffle = True,
       num_workers=2, pin_memory=True)

    train_loss = []
    val_loss = []
    val_acc = []
    mean_errors = []
    best = False
    for epoch in range(0,1):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch)
        train_loss = train_loss + loss_train
        # evaluate on validation set
        loss_val, mean_error = validate(val_loader, model, criterion)
        val_loss = val_loss + loss_val
        mean_errors += mean_error
        print(mean_errors)
        state = {
            'epoch': epoch + 1,
            'arch': "REN",
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        if (epoch > 1) :
            best = (mean_error < min(mean_errors[:len(mean_errors)-1]))
            if best:
                print("saving best performing checkpoint on val")
                save_checkpoint(state, True)

        save_checkpoint(state, False)

    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_train_loss.out",train_loss)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_val_loss.out",val_loss)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_val_acc.out",val_acc)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_mean_errors.out",mean_errors)




def train(train_loader, model, criterion, optimizer, epoch):


    # switch to train mode
    model.train()
    loss_train = []
    for i, (input, target) in enumerate(train_loader):
        stime = time.time()
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
        loss_train.append(loss.data.item())
        optimizer.step()
        np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_iteration_train_loss.out", np.asarray(loss_train))
        # measure elapsed time
        if i % print_interval == 0:
            state = {
                'iteration': i+1,
                'arch': "REN",
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            save_checkpoint(state, False, filename='checkpoints/' + str(i+1) + "_checkpoint.pth.tar")
            TT = time.time() -stime
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Time: {time:.2f}\t'.format(
                   epoch, i, len(train_loader), loss=loss.data[0], time= TT))

    return [np.mean(loss_train)]



def validate(val_loader, model, criterion):

    # switch to evaluate mode
    model.eval()

    loss_val = []
    errors = []
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):

            target = target.double()
            target = target.cuda(non_blocking=True)
            # compute output
            input = input.double()
            input = input.cuda()
            output = model(input)
            errors.append(compute_distance_error(output, target).item())
            loss = criterion(output, target)

            if i % 5 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                       i, len(val_loader), loss=loss))
            loss_val.append(loss.data.item())


    return [np.mean(loss_val)] , [np.mean(errors)]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth.tar')

def load_checkpoint(path,model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def compute_distance_error(output, target):

    error = (torch.mean((target-output).abs()))
    return error

def draw_pose(img, pose):
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)

    return img

def world2pixel(x):
    fx, fy, ux, uy = 241.42, 241.42, 160, 120
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x

if __name__ == '__main__':
    main()
    # joints =read_joints()[0].numpy()
    # center =get_center(read_depth_from_bin("data/P0/5/000000_depth.bin"))
    # depth = read_depth_from_bin("data/P8/5/000000_depth.bin")
    # center = get_center(depth)
    # depth = _crop_image(depth, center, is_debug=False)
    # torch.no_grad()
    # model = REN()
    # optimizer = torch.optim.SGD(model.parameters(), 0.005,
    #                             momentum=0.9,
    #                             weight_decay=0.0005)
    #
    # model, optimizer = load_checkpoint("a_checkpoint.pth.tar", model, optimizer)
    # model.eval()
    # depth = torch.from_numpy(depth)
    # depth = torch.unsqueeze(depth, 0)
    # depth = torch.unsqueeze(depth, 0)
    # print(depth.shape)
    # results = model(depth)
    # results = (results[0].detach().numpy()).reshape(21,3)
    # print(results)
    # joints = joints.reshape(21,3)
    # print(joints.shape)
    # joints = world2pixel(joints)
    # print(joints.shape)
    # depth = read_depth_from_bin("data/P0/5/000000_depth.bin")
    # dst = draw_pose(depth, joints)
    # img_show = (dst + 1) / 2;
    # hehe = cv2.resize(img_show, (512, 512))
    # cv2.imshow('debug', dst)
    # ch = cv2.waitKey(0)
    # if ch == ord('q'):
    #     exit(0)
