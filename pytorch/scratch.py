
def check():
    torch.no_grad()
    # torch.set_printoptions(threshold=100000)
    np.set_printoptions(threshold=np.nan)
    depth = read_depth_from_bin("data/P0/5/000000_depth.bin")
    #get centers
    center = get_center(depth)
    #get cube and resize to 96x96
    depth = _crop_image(depth, center, is_debug=False)
    # print(depth)
    assert ((depth>1).sum() == 0)
    assert ((depth<-1).sum() == 0)
    #normalize
    # depth1 = normalize(depth)
    # print(np.array_equal(depth,depth1))
    depth = (torch.from_numpy(depth))
    depth = torch.unsqueeze(depth, 0)
    depth = torch.unsqueeze(depth, 0)
    # print(depth.shape)

    depth1 = read_depth_from_bin("data/P2/5/000400_depth.bin")
    #get centers
    center = get_center(depth1)
    #get cube and resize to 96x96
    depth1 = _crop_image(depth1, center, is_debug=False)
    # print(depth)
    assert ((depth1>1).sum() == 0)
    assert ((depth1<-1).sum() == 0)
    # normalize
    # depth1 = normalize(depth)
    print(np.array_equal(depth,depth1))
    depth1 = (torch.from_numpy(depth1))
    depth1 = torch.unsqueeze(depth1, 0)
    depth1= torch.unsqueeze(depth1, 0)

    print(torch.eq(depth1,depth).all())
    model = REN()
    criterion = nn.SmoothL1Loss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.005,
                                momentum=0.9,
                                weight_decay=0.0005)

    model, optimizer = load_checkpoint("2_checkpoint.pth.tar", model, optimizer)
    model.eval()
    model = model.train(False)
    model =model.cuda()
    model =model.double()
    depth = depth.cuda()
    depth = depth.double()
    depth1 = depth1.cuda()
    depth1 = depth1.double()

    test = np.ones((96,96))
    test = (torch.from_numpy(test))
    test = torch.unsqueeze(test, 0)
    test= torch.unsqueeze(test, 0)
    test = test.cuda()
    test = test.double()


    # print(depth)
    results = model(depth)
    results1 = model(depth1)
    results2 = model(test)
    joints = read_joints()
    print(joints[0])
    print(results)
    depth = read_depth_from_bin("data/P0/5/000000_depth.bin")
    img = draw_pose(depth, results)
    cv2.imshow('result', img)
    ch = cv2.waitKey(0)
    if ch == ord('q'):
        exit(0)
