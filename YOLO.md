![image-20230210181645715](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230210181645715.png)



## YOLO数据处理流程：

Yolo的CNN网络将输入的图片分割成S×S网格，然后每个单元格负责去检测那些中心点落在该格子内的目标。

1. 基本思想是使用图像分类和定位算法(image classification and Localization algorithm)然后将算法应用到九个格子上

2. 然后需要对每个小网格定义一个5+k（k为预测类别个数）维向量的目标标签

3. YOLO算法使用的是取目标对象边框中心点的算法，即考虑边框的中心点在哪个格子中

   ![image-20230210184508360](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230210184508360.png)

每个单元格会预测B个边界框（bounding box）以及边界框的置信度（confidence score）；置信度其实包含两个方面，一是这个边界框含有目标的可能性大小，二是这个边界框的准确度。

当该边界框是背景时（即不包含目标），此时Pr(object)=0。而当该边界框包含目标时，Pr(object)=1。

边界框的准确度可以用预测框与实际框（ground truth）的IOU（intersection over union，交并比）来表征，记为IOUtruthpred

因此置信度可以定义为Pr(object)∗IOUtruthpred

每个单元格需要预测(B∗5+C)个值。如果将输入图片划分为S×S网格，那么最终预测值为S×S×(B∗5+C)大小的张量。整个模型的预测值结构如下图所示。对于PASCAL VOC数据，其共有20个类别，如果使用S=7,B=2，那么最终的预测结果就是7×7×307×7×30大小的张量。

## 非极大值抑制算法（non maximum suppression, NMS）

NMS算法主要解决的是一个目标被多次检测的问题，如图中人脸检测，可以看到人脸被多次检测，但是其实我们希望最后仅仅输出其中一个最好的预测框。首先从所有的检测框中找到置信度最大的那个框，然后挨个计算其与剩余框的IOU，如果其值大于一定阈值（重合度过高），那么就将该框剔除；然后对剩余的检测框重复上述过程，直到处理完所有的检测框。



**（1）**将所有框的得分排序，选中最高分及其对应的框：

**（2）**遍历其余的框，如果和当前最高分框的重叠面积(IOU)大于一定阈值，我们就将框删除。

**（3）**从未处理的框中继续选一个得分最高的，重复上述过程。

在NMS中，使用小的阈值Nt（如0.3），将会更多的抑制附近的检测框，从而增加了错失率；而使用大的Nt（如0.7）将会增加假正的情况。

IoU是两个区域重叠的部分除以两个区域的集合部分得出的结果.

YOLOX工程文件中的nms算法：

```python
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)
```

前处理函数：

```python
def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

```

后处理函数：

```python
def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

```

（x,y,w,h）->(x1,y1,x2,y2）转换函数：

```python
def xywh2x1y1x2y2(predictions):
    boxes = predictions[:, :4]
	scores = predictions[:, 4:5] * predictions[:, 5:]
    
	boxes_xyxy = np.ones_like(boxes)
	boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
	boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
	boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
	boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
	boxes_xyxy /= ratio
    
    return boxes_xyxy, scores
```



## 损失函数

![image-20230218144359639](C:\Users\Administrator\Desktop\CUDA编程训练营\imgs\image-20230218144359639.png)



基础误差计算方法是 平方和误差

bounding box的(x, y, w, h)的坐标预测误差。

对不同大小的bounding box预测中，相比于大box大小预测偏一点，小box大小测偏一点肯定更不能被忍受。所以在Loss中同等对待大小不同的box是不合理的。为了解决这个问题，作者用了一个比较取巧的办法，即对w和h求平方根进行回归。

bounding box的confidence预测误差

由于绝大部分网格中不包含目标，导致绝大部分box的confidence=0，所以在设计confidence误差时同等对待包含目标和不包含目标的box也是不合理的。所以在不含object的box的confidence预测误差中乘以惩罚权重λnoobj=0.5。

除此之外，同等对待4个值(x, y, w, h)的坐标预测误差与1个值的conference预测误差也不合理，所以作者在坐标预测误差误差之前乘以权重λcoord=5

分类预测误差

一个网格只预测一次类别，即默认每个网格中的所有B个bounding box都是同一类。

Loss = λcoord * 坐标预测误差 + （含object的box confidence预测误差 + λnoobj * 不含object的box confidence预测误差） + 类别预测误差

## 注意：

yolo并没有使用Relu激活函数，而是使用了leaky rectified linear激活函数



# 使用自有数据集

## 数据集结构：

VOC格式排布

在VOC这些文件夹中，我们主要用到：

**① JPEGImages文件夹：**数据集图片

**② Annotations文件夹：**与图片对应的xml文件

**③ ImageSets/Main文件夹：**将数据集分为训练集和验证集，因此产生的train.txt和val.txt。

![image-20230210192932744](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230210192932744.png)

![img](https://pic3.zhimg.com/80/v2-07c3710bced694277eeeee27c02b84fa_720w.webp)

## 修改工程文件的标签：

在yolox/data/datasets/voc_classes.py中；

![image-20230210193146228](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230210193146228.png)

## 修改数据集类别数量

修改exps/example/yolox_voc/yolox_voc_s.py中的self.num_classes

```python
def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

修改训练集：

```python
dataset = VOCDetection(
                data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
                image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )
```

修改验证集：

```python
valdataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
```

## 修改训练程序

修改yolox/exp/yolox_base.py中的self.num_classes

```python
def __init__(self):
    super().__init__()

    # ---------------- model config ---------------- #
    # detect classes number of model
    self.num_classes = 80
    # factor of model depth
    self.depth = 1.00
    # factor of model width
    self.width = 1.00
    # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
    self.act = "silu"

    # ---------------- dataloader config ---------------- #
    # set worker to 4 for shorter dataloader init time
    # If your training process cost many memory, reduce this value.
    self.data_num_workers = 4
    self.input_size = (640, 640)  # (height, width)
    # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
    # To disable multiscale training, set the value to 0.
    self.multiscale_range = 5
    # You can uncomment this line to specify a multiscale range
    # self.random_size = (14, 26)
    # dir of dataset images, if data_dir is None, this project will use `datasets` dir
    self.data_dir = None
    # name of annotation file for training
    self.train_ann = "instances_train2017.json"
    # name of annotation file for evaluation
    self.val_ann = "instances_val2017.json"
    # name of annotation file for testing
    self.test_ann = "instances_test2017.json"

    # --------------- transform config ----------------- #
    # prob of applying mosaic aug
    self.mosaic_prob = 1.0
    # prob of applying mixup aug
    self.mixup_prob = 1.0
    # prob of applying hsv aug
    self.hsv_prob = 1.0
    # prob of applying flip aug
    self.flip_prob = 0.5
    # rotation angle range, for example, if set to 2, the true range is (-2, 2)
    self.degrees = 10.0
    # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
    self.translate = 0.1
    self.mosaic_scale = (0.1, 2)
    # apply mixup aug or not
    self.enable_mixup = True
    self.mixup_scale = (0.5, 1.5)
    # shear angle range, for example, if set to 2, the true range is (-2, 2)
    self.shear = 2.0

    # --------------  training config --------------------- #
    # epoch number used for warmup
    self.warmup_epochs = 5
    # max training epoch
    self.max_epoch = 300
    # minimum learning rate during warmup
    self.warmup_lr = 0
    self.min_lr_ratio = 0.05
    # learning rate for one image. During training, lr will multiply batchsize.
    self.basic_lr_per_img = 0.01 / 64.0
    # name of LRScheduler
    self.scheduler = "yoloxwarmcos"
    # last #epoch to close augmention like mosaic
    self.no_aug_epochs = 15
    # apply EMA during training
    self.ema = True

    # weight decay of optimizer
    self.weight_decay = 5e-4
    # momentum of optimizer
    self.momentum = 0.9
    # log period in iter, for example,
    # if set to 1, user could see log every iteration.
    self.print_interval = 10
    # eval period in epoch, for example,
    # if set to 1, model will be evaluate after every epoch.
    self.eval_interval = 10
    # save history checkpoint or not.
    # If set to False, yolox will only save latest and best ckpt.
    self.save_history_ckpt = True
    # name of experiment
    self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    # -----------------  testing config ------------------ #
    # output image size during evaluation/test
    self.test_size = (640, 640)
    # confidence threshold during evaluation/test,
    # boxes whose scores are less than test_conf will be filtered
    self.test_conf = 0.01
    # nms threshold
    self.nmsthre = 0.65
    self.cache_dataset = None
    self.dataset = None
```



训练命令：

`python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 64 -c yolox_s.pth`