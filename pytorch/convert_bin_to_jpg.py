from MSRADataset import read_depth_from_bin, read_joints
import cv2

def draw_pose(img, pose):
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)

    return img


joints, keys = read_joints([8])
for index in range(int(len(joints)/2),int(len(joints))):
    person = keys[index][0]
    name = keys[index][1]
    file = '%06d' % int(keys[index][2])
    joint = joints[index]
    filename = "data/P"+str(person)+"/"+str(name)+"/"+str(file)+"_depth.bin"

    depth = read_depth_from_bin(filename)
    # print("A/"+filename.replace('bin','jpg'))
    newname= '%06d'% int(index)+".jpg"
    cv2.imwrite("A/val/"+ newname,depth)
    res = draw_pose(depth, joint.reshape(21,3))
    cv2.imwrite("B/val/"+newname,res)
    # cv2.imshow('truth', res)
    # ch = cv2.waitKey(0)
    # if ch == ord('q'):
    #     exit(0)
