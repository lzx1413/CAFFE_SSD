import json

label_dict = dict()

f = open('./labels.txt', 'r')
for i in f:
    name = i.strip().split(',')[-1]
    label = i.strip().split(',')[-2]
    label_dict[name] = label
f.close()
result = []

#predict_root = '/home/user/Program/caffe-model/det/rfcn/models/coco/resnet101-v2/ss-ohem-multigrid-context/predict_dev800/'
predict_root = '/home/super/workspace/zuoxin/CAFFE_SSD/predict_ss_coco_fpn/'
# predict_root = '/home/user/Program/caffe-model/det/rfcn/models/coco/air152/ms-ohem-multigrid-deformpsroi-multicontext/predict_test-new1-5w/'
# predict_root = '/home/user/Program/caffe-model/det/rfcn/models/person/se-resnet50/800-ohem-multigrid/predict_ss/'


for _ in label_dict.keys():
    pre_file = predict_root + 'comp4_det_test_' + _ + '.txt'
    f = open(pre_file, 'r')
    print pre_file
    for i in f:
        image_id = int(i.strip().split(' ')[0].split('.jpg')[0])
        # image_id = int(i.strip().split(' ')[0].split('_')[-1])
        category_id = int(label_dict[_])
        score = float(i.strip().split(' ')[1])
        bbox = [float(i.strip().split(' ')[2]), 
                float(i.strip().split(' ')[3]), 
                round(float(i.strip().split(' ')[4]) - float(i.strip().split(' ')[2]), 1), 
                round(float(i.strip().split(' ')[5]) - float(i.strip().split(' ')[3]), 1)]
        tmp = dict(image_id=image_id, category_id=category_id, bbox=bbox, score=score)
        result.append(tmp)
    f.close()

with open('./result.json', 'w') as www:
    json.dump(result, www)



