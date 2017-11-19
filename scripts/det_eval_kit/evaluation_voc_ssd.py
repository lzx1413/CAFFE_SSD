# -*- coding:utf-8 -*-
import sys

sys.path.append('/home/foto1/workspace/py-RFCN-priv/caffe-priv/python')
import caffe
import cv2
import os
import numpy as np
import datetime
import argparse
from PIL import Image, ImageDraw
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cPickle
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

val_file = '2012test.txt'  # for voc2007 test
image_root = '/mnt/lvmhdd1/zuoxin/dataset/VOCdevkit/VOC2012_test/JPEGImages/'
#image_root = '/mnt/lvmhdd1/zuoxin/dataset/VOCdevkit/VOC2007/JPEGImages/'
save_root = './predict/predict_ss_fssd300_voc12++/'
NUM_CLS = 21
_classes = ('__background__',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--labelmap_file',
                    default='data/VOC0712/labelmap_voc.prototxt')
#parser.add_argument('--model_def',
#                    default='/mnt/lvmhdd1/zuoxin/ssd_models/deploy_VGG_SSD300_VOC0712.prototxt')
#parser.add_argument('--model_def',
    #               default='/home/zuoxin/workspace/detections/CAFFE_SSD/ssd_models/FSSD512/deploy.prototxt')
parser.add_argument('--model_def',
                    default='/home/zuoxin/workspace/detections/CAFFE_SSD/jobs/VGGNet/SSD_FPN_COCOP_300x300/1109/deploy.prototxt')
parser.add_argument('--image_resize', default=300, type=int)
parser.add_argument('--image_save_path', default='')
parser.add_argument('--save_result',default = True)
parser.add_argument('--save_img_result',default = False)
#parser.add_argument('--model_weights',
#                    default='/home/zuoxin/workspace/detections/CAFFE_SSD/ssd_models/FSSD512/VGG_VOC0712++_SSD_FPN_COCOP_512x512_iter_60000.caffemodel')
#parser.add_argument('--model_weights',
#                    default='/mnt/lvmhdd1/zuoxin/ssd_models/VGGNet/SSD_FPN_COCOVOC_0712++_300x300/1028/VGG_VOC0712_0712++_SSD_FPN_COCOVOC_0712++_300x300_iter_80000.caffemodel')
#parser.add_argument('--model_weights',
#                    default='/mnt/lvmhdd1/zuoxin/ssd_models/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
parser.add_argument('--model_weights',
                    default='ssd_models/VGG_VOC0712_SSD_FPN_COCOP_300x300_iter_80000.caffemodel')
parser.add_argument('--image_file', default='examples/images/fish-bike.jpg')
args = parser.parse_args()
class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels

    def detect(self,image_file, conf_thresh=0.01, topn=-1):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        timer_pt1 = datetime.datetime.now()
        detections = self.net.forward()['detection_out']
        timer_pt2 = datetime.datetime.now()

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        #top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        im_boxes = [[] for _ in xrange(NUM_CLS)]
        im_boxes_for_show = []


        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            if xmin < 0:
                xmin = 0 
            if ymin < 0:
                ymin = 0 
            if xmax> image.shape[1]:
                xmax = image.shape[1] 
            if ymax > image.shape[0]:
                ymax = image.shape[0] 
            score = top_conf[i]
            label = int(top_label_indices[i])
            result = dict()
            result['label'] = label
            result['score'] = score
            result['bbox'] = [xmin,ymin,xmax,ymax]
            im_boxes_for_show.append(result)
            im_boxes[label].extend([xmin, ymin, xmax, ymax,score])
        im_boxes = [np.asarray(_).reshape((-1,5)) for _ in im_boxes ]

        return im_boxes,im_boxes_for_show, str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds)



def showResults(img_file, results,result_path, labelmap=None, threshold=0.5, display=None):
    if not os.path.exists(img_file):
        print "{} does not exist".format(img_file)
        return
    img = io.imread(img_file)
    plt.clf()
    plt.imshow(img)
    plt.axis('off');
    ax = plt.gca()
    if labelmap:
        # generate same number of colors as classes in labelmap.
        num_classes = len(labelmap.item)
    else:
        # generate 20 colors.
        num_classes = 20
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    for res in results:
 
        if 'score' in res and threshold and float(res["score"]) < threshold:
            continue
       
        label = res['label']
        name = _classes[label]
        color = colors[label % num_classes]
        bbox = res['bbox']
        coords = (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        if 'score' in res:
            score = res['score']
            display_text = '%s: %.2f' % (name, score)
        else:
            display_text = name
        ax.text(bbox[0], bbox[1], display_text, bbox={'facecolor':color, 'alpha':0.5})
        #plt.show()
        if not os.path.exists(os.path.join(save_root,'img_result')):
            os.makedirs(os.path.join(save_root,'img_result'))
        plt.savefig(result_path, bbox_inches="tight")

def eval_batch(detector,max_per_image=100000, thresh=0.0):
    eval_images = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip())

    skip_num = 0
    eval_len = len(eval_images)

    all_boxes = [[[] for _ in xrange(eval_len - skip_num)]
                 for _ in xrange(NUM_CLS)]

    start_time = datetime.datetime.now()
    t = 0
    for i in xrange(eval_len - skip_num):
        test_img = image_root + eval_images[i + skip_num] + '.jpg'

        im_boxes,im_boxes_for_show,spend_time = detector.detect(test_img)
       
        if args.save_img_result:
            save_img_path = os.path.join(save_root,'img_result',eval_images[i+skip_num]+'_result.jpg')
            showResults(test_img,im_boxes_for_show,save_img_path)
	    t = t+float(spend_time)
        print 'Testing image: {}/{} {} {}s'.format(str(i + 1), str(eval_len - skip_num), str(eval_images[i + skip_num]),spend_time)

        for j in xrange(1, NUM_CLS):
            all_boxes[j][i] = im_boxes[j]

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, NUM_CLS)])

            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, NUM_CLS):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        timer_pt3 = datetime.datetime.now()

    det_file = './detections.pkl'
    t = t/eval_len
    print 'avg time is ',t
    if not args.save_result:
        return 
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    for cls_ind, cls in enumerate(_classes):
        if cls == '__background__':
            continue
        print 'Writing {} VOC results file'.format(cls)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        filename = save_root + 'comp4' + '_det' + '_test_' + cls + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(eval_images[skip_num:]):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1], dets[k, 0] , dets[k, 1] , dets[k, 2], dets[k, 3] ))

    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))

if __name__ == '__main__':

	detection = CaffeDetection(0,
                           args.model_def, args.model_weights,
                           args.image_resize, args.labelmap_file)
	#det_result = detection.detect(args.image_file)
	eval_batch(detection)
