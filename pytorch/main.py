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
import cv2

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
    mean_errors = []
    for epoch in range(0,150):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch)
        train_loss = train_loss + loss_train
        # evaluate on validation set
        loss_val, mean_error = validate(val_loader, model, criterion)
        val_loss = val_loss + loss_val
        mean_errors += mean_error
        print(mean_errors)
        print(type(mean_error[0] ))
        print(type(mean_errors[len(mean_errors)-1]))
        if mean_error[0] < mean_errors[len(mean_errors)-1]:
            print("saving best performing checkpoint on val")
            save_checkpoint(state, True)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "REN",
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, False)

    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "_train_loss.out",train_loss)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "val_loss.out",val_loss)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "val_acc.out",val_acc)
    np.savetxt("history/"+ OUTFILE.replace(".csv", "") + "mean_errors.out",mean_errors)




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
        loss_train.append(loss.data.item())
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
        shutil.copyfile(filename, 'model_best.pth.tar')

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
        print(pt)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)
    return img

def _transform_pose(poses, centers):
    _fx, _fy, _ux, _uy = 241.42, 241.42, 160, 120
    res_poses = poses.detach().numpy() * 150
    num_joint = 21
    centers_tile = np.tile(centers.astype(int), (num_joint, 1, 1)).transpose([1, 0, 2])
    res_poses[:, 0::3] = res_poses[:, 0::3] * _fx / centers_tile[:, :, 2] + centers_tile[:, :, 0]
    res_poses[:, 1::3] = res_poses[:, 1::3] * _fy / centers_tile[:, :, 2] + centers_tile[:, :, 1]
    res_poses[:, 2::3] += centers_tile[:, :, 2]
    res_poses = np.reshape(res_poses, [poses.shape[0], -1, 3])
    return res_poses

if __name__ == '__main__':
    main()
    # torch.no_grad()
    # depth = read_depth_from_bin("data/P0/5/000000_depth.bin")
    # center = get_center(depth)
    # depth = _crop_image(depth, center, is_debug=False)
    # depth = (torch.from_numpy(depth))
    # depth = torch.unsqueeze(depth, 0)
    # depth = torch.unsqueeze(depth, 0)
    # print(center.astype(int))
    # print(depth.shape)
    # model = REN()
    # criterion = nn.SmoothL1Loss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), 0.005,
    #                             momentum=0.9,
    #                             weight_decay=0.0005)
    #
    # model, optimizer = load_checkpoint("2_checkpoint.pth.tar", model, optimizer)
    # model.eval()
    # model =model.cuda()
    # model =model.double()
    # depth = depth.cuda()
    # depth = depth.double()
    # results = model(depth)
    # joints = read_joints()
    #
    # pose = _transform_pose(results.cpu(),center)[0]
    # print(pose[0][0])
    # print(type(depth))
    # img = draw_pose(depth.cpu().numpy(), pose)
    # cv2.imshow('result', img)
    # ch = cv2.waitKey(0)
    # if ch == ord('q'):
    #     exit(0)
