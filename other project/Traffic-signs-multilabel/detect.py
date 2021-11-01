import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import PIL

color_attrs = ['red','yellow','blue','other']
type_attrs = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60',
           'speed limit 70', 'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
           'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road',
           'give way', 'stop', 'no traffic both ways',
           'no trucks', 'no entry', 'danger', 'bend left', 'bend right', 'bend', 'uneven road',
           'slippery road', 'road narrows', 'construction', 'traffic signal', 'pedestrian crossing',
           'school crossing','cycles crossing', 'snow', 'animals', 'restriction ends',
           'go right', 'go left', 'go straight', 'go right or straight','go left or straight', 'keep right',
           'keep left', 'roundabout', 'restriction ends (overtaking)', 'restriction ends (overtaking (trucks))']
direction_attrs = ['prohibitory','danger','mandatory','other']
device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')



class Cls_Net(torch.nn.Module):
    """
    vehicle multilabel classification model
    """

    def __init__(self, num_cls, input_size):
        """
        network definition
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        # delete original FC and add custom FC
        self.features = torchvision.models.resnet50(pretrained=True)
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(2048 ** 2, num_cls)  # 输出类别数
        # print('=> fc layer:\n', self.fc)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        X = self.features(X)  # extract features

        X = X.view(N, 2048, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN

        X = X.view(N, 2048 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X


# ------------------------------------- vehicle detection model
class Car_Classifier(object):
    """
    vehicle detection model mabager
    """

    def __init__(self,
                 num_cls,
                 model_path=""):
        """
        load model and initialize
        """
        # define model and load weights
        self.net = Cls_Net(num_cls=num_cls, input_size=224).to(device)
        # self.net = torch.nn.DataParallel(Net(num_cls=20, input_size=224),
        #                                  device_ids=[0]).to(device)
        self.net.load_state_dict(torch.load(model_path))
        print('=> vehicle classifier loaded from %s' % model_path)

        # set model to eval mode
        self.net.eval()

        # test data transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # split each label
        self.color_attrs = color_attrs
        print('=> color_attrs:\n', self.color_attrs)

        self.direction_attrs = direction_attrs
        print('=> direction attrs:\n', self.direction_attrs)

        self.type_attrs = type_attrs
        print('=> type_attrs:\n', self.type_attrs)

    def get_predict(self, output):
        """
        get prediction from output
        """
        # get each label's prediction from output
        output = output.cpu()  # fetch data from gpu
        pred_color = output[:, :4]
        pred_direction = output[:, 4:8]
        pred_type = output[:, 8:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def pre_process(self, image):
        """
        image formatting
        :rtype: PIL.JpegImagePlugin.JpegImageFile
        """
        # image data formatting
        if type(image) == np.ndarray:
            if image.shape[2] == 3:  # turn all 3 channels to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 1:  # turn 1 channel to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # turn numpy.ndarray into PIL.Image
            image = Image.fromarray(image)
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            if image.mode == 'L' or image.mode == 'I':  # turn 8bits or 32bits into 3 channels RGB
                image = image.convert('RGB')

        return image

    def predict(self, img):
        """
        predict vehicle attributes by classifying
        :return: vehicle color, direction and type
        """
        # image pre-processing
        img = self.transforms(img)
        img = img.view(1, 3, 224, 224)

        # put image data into device
        img = img.to(device)

        # calculating inference
        output = self.net.forward(img)

        # get result
        # self.get_predict_ce, return pred to host side(cpu)
        pred = self.get_predict(output)
        print("pred",pred[0][2])
        color_name = self.color_attrs[pred[0][0]]
        direction_name = self.direction_attrs[pred[0][1]]
        type_name = self.type_attrs[pred[0][2]]

        return color_name, direction_name, type_name

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = True
    # if classify:
    #     modelc = torch_utils.load_classifier(name='resnet50', n=1000)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet50_Best_Model18.pth', map_location=device)['model'])  # load weights
    #     modelc.to(device).eval()
    if classify:
        modelc = Car_Classifier(num_cls=51,model_path=opt.model_path)


    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy)
                    x=xyxy
                    ROI = Image.fromarray(im0s[int(x[1]):int(x[3]),int(x[0]):int(x[2])])
                    # ROI.show()
                    car_color, car_direction, car_type = modelc.predict(ROI)
                    label = str(car_color + ' ' + car_direction + ' ' + car_type)
                    print(label)

                    #将框绘制在原图里
                    # color = [random.randint(0, 255) for _ in range(3)]
                    # tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    # cv2.rectangle(im0s, c1, c2,color,thickness=tl, lineType=cv2.LINE_AA)
                    # print("det",det[:, :4])
                    # print(int(det[:,0]))
                    # # xxx=im0s[int(det[:,0]):int(det[:,2]),int(det[:,1]):int(det[:,3])]
                    # cv2.imshow("img",im0s)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow()




                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='onedata.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best_ones.pt', help='weights path  weights/yolov3-spp-ultralytics.pt')
    parser.add_argument('--model-path', type=str, default='./multilabel/checkpoints/epoch_111.pth', help='multilabel weights path')
    parser.add_argument('--source', type=str, default='data/samples/test_images', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    # print(opt)

    with torch.no_grad():
        detect()
