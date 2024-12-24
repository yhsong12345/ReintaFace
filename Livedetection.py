from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from utils.yolodetection import *
import numpy as np
from data.prior_box import PriorBox
from utils.py_cpu_nms import py_cpu_nms
import random
import cv2
from model.retinaFace import RetinaFace
from utils.box import decode, decode_landm



# def check_keys(model, pretrained_state_dict):
#     ckpt_keys = set(pretrained_state_dict.keys())
#     model_keys = set(model.state_dict().keys())
#     used_pretrained_keys = model_keys & ckpt_keys
#     unused_pretrained_keys = ckpt_keys - model_keys
#     missing_keys = model_keys - ckpt_keys
#     print('Missing keys:{}'.format(len(missing_keys)))
#     print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
#     print('Used keys:{}'.format(len(used_pretrained_keys)))
#     assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
#     return True


# def remove_prefix(state_dict, prefix):
#     ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
#     print('remove prefix \'{}\''.format(prefix))
#     f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
#     return {f(key): value for key, value in state_dict.items()}


# def load_model(model, pretrained_path):
#     print('Loading pretrained model from {}'.format(pretrained_path))
#     pretrained_dict = torch.load(pretrained_path)
#     if "state_dict" in pretrained_dict.keys():
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#     else:
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#     check_keys(model, pretrained_dict)
#     model.load_state_dict(pretrained_dict, strict=False)
#     return model


def main(args):
    torch.set_grad_enabled(False)
    m = args.network 
    # net and model
    net = RetinaFace(m)
    best_model_cp = torch.load(args.trained_model)
    net.load_state_dict(best_model_cp['model_state_dict'])
    # net = load_model(net, args.trained_model)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")
    net = net.to(device)

    # testing scale
    resize = 1

    # testing begin
    cap = cv2.VideoCapture(0)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # while True:
    #         view_img = check_imshow()
    #         cudnn.benchmark = True  # set True to speed up constant image size inference
    #         dataset = LoadStreams(cap, img_size=args.image_size)

    #         # Get names and colors
    #         model = net
    #         names = model.module.names if hasattr(model, 'module') else model.names
    #         colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            
    #         for path, img, im0s, vid_cap in dataset:
    #             img = torch.from_numpy(img).to(device)
    #             img = img.float()  # uint8 to fp16/32
    #             img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #             if img.ndimension() == 3:
    #                 img = img.unsqueeze(0)

    #             # Inference
    #             t1 = time_synchronized()
    #             with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    #                 pred = model(img)
    #             t2 = time_synchronized()

    #             # Process detections
    #             for i, det in enumerate(pred):  # detections per image
    #                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
    

    #                 p = Path(p)  # to Path
    #                 gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #                 if len(det):
    #                     # Rescale boxes from img_size to im0 size
    #                     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    #                     # Print results
    #                     for c in det[:, -1].unique():
    #                         n = (det[:, -1] == c).sum()  # detections per class
    #                         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    #                     # Write results
    #                     for *xyxy, conf, cls in reversed(det):
    #                         if view_img:
    #                             label = f'{names[int(cls)]} {conf:.2f}'
    #                             plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    #                 # Print time (inference + NMS)
    #                 print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    #             # Stream results
    #         if view_img:
    #             cv2.imshow(str(p), im0)
    #             cv2.waitKey(1)  # 1 millisecond

    #         if cv2.waitKey(1) & 0xFF ==ord('q'):
    #             cap.release()
    #             break



    while cap.isOpened():
        ret, frame = cap.read()
        img_raw = frame
        variance = [0.1, 0.2]
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = img - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float().unsqueeze(0) 
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, variance)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, variance)
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold) 

        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        image_no_detection = img_raw.copy()
        # show image
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            print(b)
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        
        # show video
        cv2.imshow('Result of ArcFace', cv2.resize(image_no_detection, (800, 600)))
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            cap.release()
            break
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='./outputs/mobilenet0.25/best_model.pt',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobilenet0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--confidence_threshold', default=0.7, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
    parser.add_argument('-s','--image_size', default=640, type=int)
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    args = parser.parse_args()
    main(args)