import cv2
import warnings
warnings.simplefilter("ignore")
from MSRADataset import MSRADataset, read_depth_from_bin, get_center, _unnormalize_joints
from REN import REN
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from loss import Modified_SmoothL1Loss
import numpy as np
import time
import argparse
import datetime
import os, sys
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Region Ensemble Network')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epoch', type=int, default=40, help='number of epochs')
parser.add_argument('--test', action='store_true', help='only test without training')
parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='decay lr by 10 after _ epoches')
parser.add_argument('--input_size', type=int, default=96, help='decay lr by 10 after _ epoches')
parser.add_argument('--num_joints', type=int, default=42, help='decay lr by 10 after _ epoches')
parser.add_argument('--no_augment', action='store_true', help='dont augment data?')
parser.add_argument('--no_validate', action='store_true', help='dont validate data when training?')
parser.add_argument('--augment_probability', type=float, default=1.0, help='augment probability')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay')
parser.add_argument('--poses', type=str, default=None, nargs='+', help='poses to train on')
parser.add_argument('--persons', type=str, default=None, nargs='+', help='persons to train on')
parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')
parser.add_argument('--print_interval', type=int, default=500, help='print interval')
parser.add_argument('--save_dir', type=str, default="experiments/", help='path/to/save_dir')
parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models. if none, it will be saved as the date and time')
parser.add_argument('--finetune', action='store_true', help='use a pretrained checkpoint')


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    expr_dir = os.path.join(opt.save_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every args.lr_decay epochs"""
    # lr = 0.00005
    lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    # print("LR is " + str(lr)+ " at epoch "+ str(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def set_default_args(args):
    if not args.name:
        now = datetime.datetime.now()
        args.name = now.strftime("%Y-%m-%d-%H-%M")
    if not args.poses:
        args.poses = ["1","2","3","4",'5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
    if not args.persons:
        args.persons = [0,1,2,3,4,5,6,7]

    args.augment = not args.no_augment
    args.validate = not args.no_validate



def main(args):
    set_default_args(args)
    model = REN(args)

    model.float()
    model.cuda()
    model.apply(weights_init)
    cudnn.benchmark = True
    criterion = Modified_SmoothL1Loss().cuda()

    train_dataset = MSRADataset(training = True, augment = args.augment, args = args)
    test_dataset = MSRADataset(training = False, augment= False, args = args)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=args.batchSize, shuffle = True,
       num_workers=0, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=args.batchSize  ,shuffle = True,
       num_workers=0, pin_memory=False)

    current_epoch = 0
    if args.checkpoint:
        model, optimizer, current_epoch = load_checkpoint(args.checkpoint, model, optimizer)
        if args.finetune:
            current_epoch = 0

    if args.test:
        test(model, args)
        return

    train_loss = []
    val_loss = []
    best = False

    print_options(args)
    expr_dir = os.path.join(args.save_dir, args.name)

    for epoch in range(current_epoch, args.epoch):

        optimizer = adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, epoch, args)
        train_loss = train_loss + loss_train
        if args.validate:
            # evaluate on validation set
            loss_val = validate(val_loader, model, criterion ,args)
            val_loss = val_loss + loss_val

        state = {
            'epoch': epoch,
            'arch': "REN",
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        if not os.path.isfile(os.path.join(expr_dir, 'model_best.pth.tar')):
            save_checkpoint(state, True, args)

        if (args.validate) and (epoch > 1) :
            best = (loss_val < min(val_loss[:len(val_loss)-1]))
            if best:
                print("saving best performing checkpoint on val")
                save_checkpoint(state, True, args)

        save_checkpoint(state, False, args)
    #

    expr_dir = os.path.join(args.save_dir, args.name)
    np.savetxt(os.path.join(expr_dir, "train_loss.out"),train_loss, fmt='%f')
    save_plt(train_loss, "train_loss")
    np.savetxt(os.path.join(expr_dir, "val_loss.out"),val_loss, fmt='%f')
    save_plt(val_loss, "val_loss")







def train(train_loader, model, criterion, optimizer, epoch,args):

    # switch to train mode
    model.train()
    loss_train = []
    expr_dir = os.path.join(args.save_dir, args.name)
    for i, (input, target) in enumerate(train_loader):

        stime = time.time()
        # measure data loading time
        target = target.float()
        target = target.cuda(non_blocking=False)
        input = input.float()
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
        if i % args.print_interval == 0:
            TT = time.time() -stime
            print('epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Time: {time:.2f}\t'.format(
                   epoch, i, len(train_loader), loss=loss.item(), time= TT))

    return [np.mean(loss_train)]



def validate(val_loader, model, criterion, args):

    # switch to evaluate mode
    model.eval()

    loss_val = []
    with torch.no_grad():
        expr_dir = os.path.join(args.save_dir, args.name)

        for i, (input, target) in enumerate(val_loader):
            target = target.float()
            target = target.cuda(non_blocking=False)
            # compute output
            input = input.float()
            input = input.cuda()
            output = model(input)
            loss = criterion(output, target)

            if i % args.print_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                       i, len(val_loader), loss=loss))
            loss_val.append(loss.data.item())


    return [np.mean(loss_val)]

def test(model, args):

    # switch to evaluate mode
    model.eval()
    test_dataset = MSRADataset(training = False, augment= False, args = args)
    errors = []
    MAE_criterion = nn.L1Loss()
    with torch.no_grad():
        expr_dir = os.path.join(args.save_dir, args.name)

        input_size= args.input_size
        for i, (input, target) in enumerate(test_dataset):
            target = target.float()
            target = target.numpy().reshape(21,2)
            tmp = np.zeros((21,3))
            for j in range(len(target)):
                tmp[j,:2] = target[j]
            # compute output
            input = input.float()
            input = input.cuda()
            input = input.unsqueeze(0)
            output = model(input)
            output = output.cpu().numpy().reshape(21,2)
            tmp1 = np.zeros((21,3))
            for j in range(len(output)):
                tmp1[j,:2] = output[j]
            center = test_dataset.get_center(i)
            # errors.append(compute_distance_error(_unnormalize_joints(tmp1,center,input_size), _unnormalize_joints(tmp,center,input_size)).item())
            output = torch.from_numpy(_unnormalize_joints(tmp1,center,input_size))
            target = torch.from_numpy(_unnormalize_joints(tmp,center,input_size))
            MAE_loss = MAE_criterion(output, target)

            errors.append(MAE_loss.item())

            if i % args.print_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}\t'.format(
                       i, len(test_dataset), loss=errors[-1]))

        errors = np.mean(errors)
        print(errors)
        if "model_best" in args.checkpoint:
            np.savetxt(os.path.join(expr_dir, "average_MAE_model_best_"+args.poses[0]), np.asarray([errors]), fmt='%f')
        else:
            np.savetxt(os.path.join(expr_dir, "average_MAE_checkpoint"+args.poses[0]), np.asarray([errors]), fmt='%f')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)

def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar'):
    expr_dir = os.path.join(opt.save_dir, opt.name)
    torch.save(state, os.path.join(expr_dir, filename))
    if is_best:
        torch.save(state, os.path.join(expr_dir, 'model_best.pth.tar'))

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch  = checkpoint['epoch']

    return model, optimizer, epoch

def compute_distance_error(output, target):

    error = (np.mean(np.abs(target-output)))
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

def save_plt(array, name):
    plt.plot(array)
    plt.xlabel('epoch')
    plt.ylabel('name')
    plt.savefig(name+'.png')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
