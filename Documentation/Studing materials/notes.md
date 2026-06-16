# Table of contents

- [Week 05 - Object Detection](#week-05---object-detection)
  - [Adding new layers to an already instantiated model](#adding-new-layers-to-an-already-instantiated-model)
  - [Leveraging pre-trained models](#leveraging-pre-trained-models)
  - [Object detection](#object-detection)
    - [The context around object detection](#the-context-around-object-detection)
    - [Drawing bounding boxes](#drawing-bounding-boxes)
    - [Detecting multiple objects](#detecting-multiple-objects)
      - [Problem - multiple objects](#problem---multiple-objects)
      - [Problem - different scales](#problem---different-scales)
      - [Problem - different aspect ratios](#problem---different-aspect-ratios)
  - [Non-maximum suppression (NMS)](#non-maximum-suppression-nms)
  - [Intersection over union (IoU)](#intersection-over-union-iou)
  - [Rich feature hierarchies for accurate object detection and semantic segmentation (`R-CNN`)](#rich-feature-hierarchies-for-accurate-object-detection-and-semantic-segmentation-r-cnn)
  - [YOLO](#yolo)
    - [Training](#training)
    - [Architecture](#architecture)
    - [Loss](#loss)
  - [Evaluating object detection models](#evaluating-object-detection-models)

# Week 05 - Object Detection

## Adding new layers to an already instantiated model

Let's do a very brief refresher on how datasets work in PyTorch.

We have the following image dataset:

```python
train_dataset
```

```console
Dataset ImageFolder
    Number of datapoints: 202
    Root location: /home/repl/pets_data/train
    StandardTransform
Transform: ToTensor()
```

<details>
<summary>How can we determine the number of classes in this dataset?</summary>

We can go through the whole dataset and create a `set` with all the second elements (as each element is a `tuple`) and then take its length:

```python
{label for _, label in train_dataset}
```

```console
{0, 1}
```

</details>

To add new layers to an already instantiated model we can use the `add_module` method. It takes a module object and adds the layer as the last one in that object.

```python
import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

net = Net()
conv2 = nn.Conv2d(in_channels=net.conv1.out_channels, out_channels=32, kernel_size=3, padding=1)
net.add_module('conv2', conv2)
print(net)
```

```console
Net(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```

The above technique is helpful when we want to see how changing an already established architecture a little bit would affect the overall model performance.

To create submodules in a class definition we can use the class [nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#modulelist):

```python
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

This, on the other hand, is helpful when we want to add multiple layers in bulk. It's like using `nn.Sequential`, but with control over the intermediate values `x`.

By the way let's ponder a little bit about the `for` loop above - do you understand what is happening there?

It's actually an example of the so-called skip-connections or residual connections. It shows how we can combine (with a summation operation) the output of the current layer with the output of a previous layer.

Here's what we get with the above 10 layers:

```text
x0
->  x1 = [L0(x0) + L0(x0)] # i=0 => 0//2=0
->  x2 = [L0(x1) + L1(x1)] # i=1 => 1//2=0
->  x3 = [L1(x2) + L2(x2)] # i=2 => 2//2=1
->  x4 = [L1(x3) + L3(x3)] # i=3 => 3//2=1
->  x5 = [L2(x4) + L4(x4)] # i=4 => 4//2=2
->  x6 = [L2(x5) + L5(x5)] # i=5 => 5//2=2
->  x7 = [L3(x6) + L6(x6)] # i=6 => 6//2=3
->  x8 = [L3(x7) + L7(x7)] # i=7 => 7//2=3
->  x9 = [L4(x8) + L8(x8)] # i=8 => 8//2=4
-> x10 = [L4(x9) + L9(x9)] # i=9 => 9//2=4
We then return x10.
```

## Leveraging pre-trained models

<details>
<summary>What are two problems with training large models from scratch?</summary>

- It's a long process (days/weeks/maybe even months).
- Requires lots of:
  - data;
  - hardware resources.

</details>

<details>
<summary>How can we solve these problems?</summary>

We can often use **pre-trained models**: models that are already trained on a task:

- Can be applied directly on the same task or a similar one, so long as it's not that different. It's our responsibility to decide what "different enough" means, but generally it means working with almost the same class labels and similar domain.
- If the task is completely different, we'd need to train the models for (usually) a few epochs to adjust their weights to the new task (a process also known as **transfer learning**).

</details>

<details>
<summary>Do you know how to access popular pre-trained vision models via PyTorch?</summary>

In the package `torchvision`! Explore available models for classification in [the torchvision model zoo](https://pytorch.org/vision/stable/models.html#classification).

</details>

The package makes it very easy to load and save models locally.

```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT # details here: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
model = resnet18(weights=weights)
transforms = weights.transforms()
print(model)
```

```console
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    ...
  )
  ...
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

Generating a new prediction is straightforward since the model does not need training:

```python
model.eval()

with torch.no_grad():
  pred = model(input_image).squeeze(0) # remove batch dimension

pred_cls = pred.softmax(0)
cls_id = pred_cls.argmax().item()
cls_name = weights.meta['categories'][cls_id]

print(cls_name)
```

We can save the parameters of any model that inherits the class `nn.Module` using the function [`torch.save`](https://pytorch.org/docs/main/generated/torch.save.html):

- Model extension: `.pt`.
- Save model weights and biases with `.state_dict()`.

```python
torch.save(model.state_dict(), 'BinaryCNN.pt')
```

Loading PyTorch models is also easy, though note that usually, loading a model is actually interpreted as "loading the weights for an instantiated model class", i.e. we have two steps:

1. Instantiate a new model (we must have the class definition):

```python
new_model = BinaryCNN()
```

2. Load saved parameters:

```python
new_model.load_state_dict(torch.load('BinaryCNN.pt'))
```

## Object detection

### The context around object detection

<details>
<summary>What is the goal of object classification?</summary>

Assign one **or multiple** class labels to a single image from a set of predefined class categories defined in the training dataset.

![w05_im_cls.png](assets/w05_im_cls.png "w05_im_cls.png")

</details>

<details>
<summary>What is the goal of object localization?</summary>

Single label classification + localization of object.

Classify the object present in an image and return its _bounding box_.

![w05_im_cls_loc.png](assets/w05_im_cls_loc.png "w05_im_cls_loc.png")

</details>

<details>
<summary>What is the goal of object detection?</summary>

Locate all instances of all object classes in an image.

![w05_im_cls_loc_det.png](assets/w05_im_cls_loc_det.png "w05_im_cls_loc_det.png")

</details>

<details>
<summary>Give at least three usecases where object detection might be useful.</summary>

- **Surveillance.** <-- most popular
- Medical diagnosis.
- Traffic management.
- Sports analytics.

</details>

<details>
<summary>So, what is the output of object detection?</summary>

- Location of each object in the image (also known as **bounding box**).
- Class label of each object.
- Optionally, a confidence score is returned. It indicates how confident the model is that the object is present in the bounding box.

The models, thus, would output tuples with the set of bounding boxes and their corresponding labels.

</details>

### Drawing bounding boxes

<details>
<summary>What is a bounding box?</summary>

A rectangular box describing the object's spatial location.

![w05_od_output.png](assets/w05_od_output.png "w05_od_output.png")

</details>

<details>
<summary>How would a bounding box be implemented?</summary>

It's often represented as a tuple with `4` elements: `(x1, y1, x2, y2)`.

The elements `x1` and `y1` are typically the top-left pixel coordinates of the bounding box.

The elements `x2` and `y2` can represent different values:

- they can be the width and height of the bounding box (in pixels);
- or they can be the bottom-right pixel coordinates of the bounding box (the way PyTorch uses them).

![w05_od_example_cat.png](assets/w05_od_example_cat.png "w05_od_example_cat.png")

</details>

<details>
<summary>If we had multiple bounding boxes over an object how would we choose the one to return to the user?</summary>

We choose the tightest one:

- it should cover the entirety of the object;
- while keeping the box area as small as possible.

</details>

PyTorch has a built-in utility function to do draw bounding boxes: [`draw_bounding_boxes`](https://pytorch.org/vision/main/generated/torchvision.utils.draw_bounding_boxes.html#draw-bounding-boxes).

```python
from torchvision import utils

# Create a tensor for a single bounding box and add a dimension for number of boxes (N=1)
bbox = torch.tensor([x_min, y_min, x_max, y_max]).unsqueeze(0)

# width => width of lines forming the rectangle
# colors => what color the lines should be
# NOTE: Also takes an optional parameter "labels" that adds a label to the bounding box
bbox_image = utils.draw_bounding_boxes(image_tensor, bbox, width=3, colors='red')
```

Which of the following statements about object detection and bounding boxes is true?

```text
A. Bounding boxes are represented by the `x` and `y` coordinates of each of the four box corners, so eight numbers altogether.
B. Object detection's goal is only to identify the spatial location of an object within the image.
C. Bounding boxes are both the way to annotate the training data for object detection tasks as well as the outputs of the models.
```

<details>
<summary>Reveal answer</summary>

C.

For A: Two coordinates or one coordinate with width and height is enough.
For B: Next to locating the object within the image, object detection is also concerned with classifying the object it detected.

</details>

### Detecting multiple objects

<details>
<summary>What was the blueprint neural network architecture that was used to solve object classification problems?</summary>

1. Feature extraction.
2. Feature classification.

![w05_im_cls_arch.png](assets/w05_im_cls_arch.png "w05_im_cls_arch.png")

</details>

<details>
<summary>How can we extend the blueprint neural network architecture for object classification to facilitate solving object localization problems?</summary>

- We would have a second output (**multi-output** model) for the bounding box.
- This second output would be produced by a second set of fully-connected layers.

![w05_im_loc_arch.png](assets/w05_im_loc_arch.png "w05_im_loc_arch.png")

</details>

<details>
<summary>What loss function would we use for this new output?</summary>

MSE or RMSE.

</details>

#### Problem - multiple objects

But what if we had multiple objects that we have to return - our network can only output one bounding box?

![w05_multple_obj.png](assets/w05_multple_obj.png "w05_multple_obj.png")

<details>
<summary>Reveal answer</summary>

We can use the sliding window approach.

1. Preprocess the dataset into many different small patches of the same size.
2. Do image classification on every single patch.

![w05_multple_obj_sol.png](assets/w05_multple_obj_sol.png "w05_multple_obj_sol.png")

You can see that this does not really solve the problem, right?

</details>

#### Problem - different scales

We can also have objects of different sizes:

![w05_multple_obj_diff_scale.png](assets/w05_multple_obj_diff_scale.png "w05_multple_obj_diff_scale.png")

<details>
<summary>How would we solve this?</summary>

We can either:

- have cropping windows of different sizes:

![w05_multple_obj_diff_scale_sol1.png](assets/w05_multple_obj_diff_scale_sol1.png "w05_multple_obj_diff_scale_sol1.png")

- have one cropping window, but start up- and downscaling the images:

![w05_multple_obj_diff_scale_sol2.png](assets/w05_multple_obj_diff_scale_sol2.png "w05_multple_obj_diff_scale_sol2.png")

This also leads to efficiency problems.

</details>

#### Problem - different aspect ratios

But then, we should also be able to handle multiple types of objects:

![w05_multple_obj_diff_asp_ratio.png](assets/w05_multple_obj_diff_asp_ratio.png "w05_multple_obj_diff_asp_ratio.png")

So you can see how the complexity of these problems quickly blows up. We can roughly say that for **a single image**:

```python
computational_cost = num_scales * num_aspect_ratios * width * height
```

And this is only for a single image!

## Non-maximum suppression (NMS)

<details>
<summary>What have you heard of NMS?</summary>

Object detection models may generate many bounding boxes and some of them may be overlapping near-duplicates.

![w05_nms_example2.png](assets/w05_nms_example2.png "w05_nms_example2.png")

NMS is a technique to select the most relevant bounding boxes:

- **Non-max**: Pick the box with highest probability and discard other boxes.
- **Suppression**: Discard all boxes that have high overlap with the box with most confidence.

![w05_nms_example.png](assets/w05_nms_example.png "w05_nms_example.png")

</details>

In PyTorch we can use the function [`ops.nms`](https://pytorch.org/vision/0.20/generated/torchvision.ops.nms.html#torchvision.ops.nms):

- **Boxes**: tensors with the bounding box coordinates of the shape $[N, 4]$.
- **Scores**: tensor with the confidence score for each box of the shape $[N]$.
- **iou_threshold**: the threshold between $0.0$ and $1.0$.
- Output: indices of filtered bounding boxes.

```python
from torchvision import ops

box_indices = ops.nms(
  boxes=boxes,
  scores=scores,
  iou_threshold=0.5,
)
box_indices
```

```console
tensor([ 0,  1,  2,  8])
```

```python
filtered_boxes = boxes[box_indices]
```

## Intersection over union (IoU)

<details>
<summary>Can you infer what IoU calculates to evaluate the quality of object detection models?</summary>

- **Object of interest**: object in image we want to detect (e.g. dog).
- **Ground truth box**: the accurate bounding box around the object of interest.
- **Intersection over Union**: Measure the overlap between predicted bounding box and ground-truth bounding box.

![w05_iou_example.png](assets/w05_iou_example.png "w05_iou_example.png")

<details>
<summary>If IoU is 0, this means that ...</summary>

there is no overlap; the model is bad.

</details>

<details>
<summary>If IoU is 1, this means that ...</summary>

there is perfect overlap; the model is perfect.

Often $0.5$ is used as a threshold, i.e. if IoU >= 0.5, this is regarded as a correct prediction of a bounding box.

</details>

</details>

<details>
<summary>Knowing this is IoU a metric or part of the loss function?</summary>

IoU is a metric.

</details>

We can calculate the IoU in PyTorch using a built-in function: [`ops.box_iou`](https://pytorch.org/vision/0.20/generated/torchvision.ops.box_iou.html#torchvision.ops.box_iou).

```python
import torch
from torchvision import ops

bbox1 = torch.tensor([50, 50, 150, 150]).unsqueeze(0)
bbox2 = torch.tensor([100, 100, 200, 200]).unsqueeze(0)

print(ops.box_iou(bbox1, bbox2))
```

```console
tensor([[0.1429]])
```

## Rich feature hierarchies for accurate object detection and semantic segmentation (`R-CNN`)

- Part of [the region-based CNN family](https://en.wikipedia.org/wiki/Region_Based_Convolutional_Neural_Networks).
  - November 2013: R-CNN: <https://arxiv.org/abs/1311.2524>;
  - April 2015: Fast R-CNN: <https://arxiv.org/abs/1504.08083>;
  - June 2015: Faster R-CNN: <https://arxiv.org/abs/1506.01497>.

Up until 2016 the best model to solve all of the above problems was **Faster R-CNN**. I strongly suggest you go and read the history leading up to it, but here's a quick recap: it uses a **two-stage approach** to do object detection:

1. Use convolutional layers (backbone) to obtain lots of feature maps and get bounding box proposals from them.
2. Apply a classifier and a regressor to produce predictions.

![w05_faster_rcnn_architecture.png](assets/w05_faster_rcnn_architecture.png "w05_faster_rcnn_architecture.png")

The Region Proposal Network (RPN) was a small fully convolutional network. It had three modules: `Anchor generator`, `Region of interest (RoI) pooling`, and `Two prediction heads`:

![w05_rpn.png](assets/w05_rpn.png "w05_rpn.png")

It slided over the backbone feature map using a set of `k=9` predefined anchor boxes:

- `3` scales: 128x128, 256x256, 512x512.
  - for each of the scales, they used `3` aspect ratios: 1:1, 1:2, 2:1.

During training, the RPN:

1. Applies a small convolution (e.g., `3x3`) over the backbone feature map to produce an intermediate feature representation.
2. At each spatial location, for `k=9` anchor boxes, it predicts:

- `k` objectness scores: one per anchor, indicating how confident the network is that there is an object there.
- `4k` bounding box regression values: four offsets per anchor - `t_x`, `t_y`, `t_w`, `t_h`.

The regression offsets adjust each anchor box to better fit a ground-truth object:

- (`t_x`, `t_y`) shift the anchor’s center relative to its width and height;
- (`t_w`, `t_h`) scale the anchor’s width and height.

After predicting objectness scores and box refinements, the RPN applies NMS to remove highly overlapping proposals and keep only the top-scoring ones.

The filtered proposals are then passed to the second stage, where region-of-interest (RoI) pooling extracts fixed-size features and a classifier + regressor produce the final object predictions. The RoI Pooling module extracts features for each proposal from the convolutional feature map and resizes them to a fixed 7×7×D size. The choice of 7×7 comes from the input size expected by the VGG16 fully connected layers.

![w05_roi_pool.png](assets/w05_roi_pool.png "w05_roi_pool.png")

We can create our own custom FasterRCNN like so:

```python
from torchvision.models.detection import rpn
from torchvision import ops
from torchvision.models.detection import FasterRCNN

anchor_generator = rpn.AnchorGenerator(
  sizes=((32, 64, 128), ),
  aspect_ratios=((0.5, 1.0, 2.0), ),
)

roi_pooler = ops.MultiScaleRoIAlign( # https://pytorch.org/vision/main/generated/torchvision.ops.MultiScaleRoIAlign.html#multiscaleroialign
  featmap_names=['0'], # the first layer labeled '0' in our backbone architecture
  output_size=7,
  sampling_ratio=2, # how many samples are taken from each bin when pooling
)

backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
backbone.out_channels = 1280

model = FasterRCNN(
  backbone=backbone,
  num_classes=num_classes,
  rpn_anchor_generator=anchor_generator,
  box_roi_pool=roi_pooler,
)
```

We can also use a pre-trained `Faster R-CNN` without manually extracting a backbone from a different model:

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 5

model = torchvision.models.detection.faster_rcnn_resnet50_fpn(weights='DEFAULT')
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

## YOLO

Paper link: <https://arxiv.org/pdf/1506.02640>. This is a fantastically written paper! There aren't many of this high quality in these days. Use it as a template / guideline when developing your own Masters thesis.

<details>
<summary>We are about to open a paper - what is the process or reading a deep learning paper?</summary>

1. Look at every picture/figure - if you don't understand them, skip reading the paper. If you understand them, then continue with answers to the following questions.
2. What did the authors try to accomplish? ...one sentence summarizing the main contribution...
3. What were the key elements of the approach? ...concrete steps to reproduce the contribution...
4. What can you use yourself? ...any notes that are worth taking/perspective shifts...
5. What other references do you want to follow? ...recommended reading for more context...
6. What are some limitations? ...points that should be mentioned as nothing is perfect...

</details>

<details>
<summary>Open the paper - what does YOLO mean?</summary>

We can see it from the title of the paper: `You Only Look Once: Unified, Real-Time Object Detection`.

</details>

<details>
<summary>When did the paper come out?</summary>

On the side panel, we can see the publication date - `09.05.2016`:

![w05_date.png](assets/w05_date.png "w05_date.png")

</details>

Note that the YOLO models form a series. Today we're looking at YOLOv1, but there've many models since. They've gone from YOLOv1 to YOLO11 and the latest one today is YOLO26.

- Check out the github of the developers (`ultralytics`): <https://github.com/ultralytics/ultralytics?tab=readme-ov-file#python>.
  - To use the models from `ultralytics`, you'll have to install their package: `pip install ultralytics`.
  - Unfortunately, only the fifth iteration of the YOLO models (`YOLOv5`) is part of the `torchvision` model zoo:

    ```python
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    results = model("image.jpg")
    results.print()
    results.save()
    ```

- History from `v1` to `26`: <https://www.preprints.org/manuscript/202602.1844>.

### Training

<details>
<summary>Read the abstract - what differentiating factor do the authors point out between their model and previous work?</summary>

Their detection pipeline is a single network. It can be optimized end-to-end directly on detection performance.

In other words, they create a **One Stage** model: it directly makes predictions for multiple categories using a single network.

</details>

<details>
<summary>The authors also point out two benefits and one drawback of their approach - what are the benefits?</summary>

- Reduced complexity since we have "bounding boxes and class probabilities directly from full images in one evaluation."
- The execution is a lot faster than the approaches at the time, especially their smaller version - Fast YOLO.

</details>

<details>
<summary>What is one drawback?</summary>

"Compared to state-of-the-art detection systems, YOLO makes more localization errors".

This is actually the famous tradeoff between speed and quality of predictions. We can it even today with the small and large GPTs.

</details>

<details>
<summary>How is object detection modelled in YOLO?</summary>

As a **regression problem** to spatially separated bounding boxes and associated class probabilities.

</details>

<details>
<summary>Which figure details the inner working of the model?</summary>

`Figure 2`:

![w05_fig2.png](assets/w05_fig2.png "w05_fig2.png")

</details>

<details>
<summary>What is the idea behind 5 in the tensor size formula?</summary>

![w05_fig2_five.png](assets/w05_fig2_five.png "w05_fig2_five.png")

</details>

<details>
<summary>What values do the authors use for "S", "B", and "C"?</summary>

![w05_fig2_vals.png](assets/w05_fig2_vals.png "w05_fig2_vals.png")

</details>

<details>
<summary>And which is the dataset on which the model was evaluated?</summary>

The Pascal VOC (VOC = Visual Object Classes).

- This is the official homepage of the dataset: <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/>.
- Ultralytics also have their own page detailing how to use the dataset (downloading included): <https://docs.ultralytics.com/datasets/detect/voc/#dataset-yaml>.
  - When you scroll down to the section `Usage` there is a sample Python code for training their latest model. Upon running the code, the dataset will be automatically downloaded.

</details>

Ok - awesome! We now have the data and can start examining the training pipeline more carefully.

<details>
<summary>What is the first step of the YOLO training pipeline given a single image?</summary>

Divide the image into SxS grid cells.

![w05_fig2_first.png](assets/w05_fig2_first.png "w05_fig2_first.png")

</details>

<details>
<summary>How is the section that holds that information both as an image form and in text form called?</summary>

`2. Unified Detection`: see the second paragraph.

</details>

<details>
<summary>What happens after the image is divided into a grid?</summary>

![w05_sec2_obj.png](assets/w05_sec2_obj.png "w05_sec2_obj.png")

</details>

<details>
<summary>So in other words, each target object is assigned to the cell that ...fill-in missing part... ?</summary>

Each target object is assigned to the cell that contains that object's center.

</details>

Ok - let's say we have this image:

![w05_sec2_checkp.png](assets/w05_sec2_checkp.png "w05_sec2_checkp.png")

<details>
<summary>Which gridcell(s) would we assign the class "person"? What about the class "car"?</summary>

- Person => cell 6.
- Car => cell 5.

![w05_sec2_checkp_answ.png](assets/w05_sec2_checkp_answ.png "w05_sec2_checkp_answ.png")

Note that only one and exactly one grid cell contains a single object.

</details>

<details>
<summary>What happens after that?</summary>

![w05_sec2_pred_logic.png](assets/w05_sec2_pred_logic.png "w05_sec2_pred_logic.png")

</details>

<details>
<summary>What are the two characteristics that are captured in the returned confidence?</summary>

1. Confidence of the model that the box indeed contains an object ($Pr(\text{Object})$).
2. How accurate / good fit the predicted box is for the object it contains ($\text{IOU}^{\text{truth}}_{\text{pred}}$).

</details>

The last sentence from the above answer describes what the goal of the predicted bounding boxes is in relation to the ground truth boxes.

<details>
<summary>What is it?</summary>

The training process dictates that the box predictions of each cell should be as close as possible to the target's box assigned to that cell.

So if the model initially predicts:

![w05_sec2_ex1.png](assets/w05_sec2_ex1.png "w05_sec2_ex1.png")

Then the loss would be high and would push the model to predict the target ones:

![w05_sec2_checkp_answ.png](assets/w05_sec2_checkp_answ.png "w05_sec2_checkp_answ.png")

This should be intuitive, but please take time to understand the paragraph and if there're questions, ask!

</details>

The way YOLO predicts bounding boxes is different from what we've seen so far. You already showed that we return `5` predictions (it's not a bad idea to revisit that paragraph while answering the following questions), but let's look at the predicted width and height.

<details>
<summary>What is the relationship between the predicted width/height and image width/height?</summary>

The `w` and `h` are "relative the whole image".

</details>

<details>
<summary>Yes, so what does w=1 mean for example?</summary>

`w=1` would mean that the width of the predicted bounding box is equal to the width of whole image.

</details>

<details>
<summary>Ok - and how should the predicted "x" and "y" be interpreted?</summary>

They correspond to the center of the predicted bounding box, but note that their value is relative to the grid cell that holds that center.

</details>

So if we had this predicted bounding box:

![w05_sec2_pred_chkp.png](assets/w05_sec2_pred_chkp.png "w05_sec2_pred_chkp.png")

Could you draw where `x` and `y` would be and where the point `(0, 0)` is for them?

<details>
<summary>Reveal answer</summary>

`x` and `y` are relative to the top left corner of the grid cell and are also normalized to be between `0` and `1`.

![w05_sec2_pred_chkp_ans.png](assets/w05_sec2_pred_chkp_ans.png "w05_sec2_pred_chkp_ans.png")

![w05_sec2_chkp_answ.png](assets/w05_sec2_chkp_answ.png "w05_sec2_chkp_answ.png")

</details>

Ok, so let's say the model returns two bounding box predictions (in blue) for one cell, i.e.:

![w05_sec2_multiple_boxes_per_cell.png](assets/w05_sec2_multiple_boxes_per_cell.png "w05_sec2_multiple_boxes_per_cell.png")

<details>
<summary>Continue reading the section - how does the model decide which one to use for the final prediction for this grid cell?</summary>

The box with the highest IOU is considered to be the prediction of the grid cell and only it will be trained to match the target one.

![w05_sec22_multi_box_answ.png](assets/w05_sec22_multi_box_answ.png "w05_sec22_multi_box_answ.png")

</details>

### Architecture

<details>
<summary>Which figure shows the architecture of YOLO?</summary>

![w05_fig3.png](assets/w05_fig3.png "w05_fig3.png")

</details>

<details>
<summary>How do the authors train this model?</summary>

They replace the inception modules used by GoogLeNet with `1x1` convolution layers followed by `3x3` convolutional layers:

![w05_fig3_1.png](assets/w05_fig3_1.png "w05_fig3_1.png")

- You can checkout the GoogLeNet paper here: <https://arxiv.org/pdf/1409.4842>.
  - The inception modules in question are the non-linear, branching pipelines in Figure 3 of the paper:

![w05_fig3_inception_modules.png](assets/w05_fig3_inception_modules.png "w05_fig3_inception_modules.png")

They then pretrain this network on the ImageNet 2012 dataset for which the input size of the images is `224x224`:

![w05_pretraining.png](assets/w05_pretraining.png "w05_pretraining.png")

After pretraining they:

- add four convolutional layers `3x3`;
- add two fully connected layers;
- increase the input size to `448x448`.

![w05_posttraining.png](assets/w05_posttraining.png "w05_posttraining.png")

</details>

<details>
<summary>What activation function was used in the hidden layers?</summary>

Leaky relu with `negative_slope=0.1`:

![w05_hidden_act_function.png](assets/w05_hidden_act_function.png "w05_hidden_act_function.png")

</details>

<details>
<summary>Look at the figure with the architecture again - can point exactly which are the newly added convolutional layers?</summary>

![w05_posttraining_new.png](assets/w05_posttraining_new.png "w05_posttraining_new.png")

</details>

### Loss

<details>
<summary>What is the name of the section that contains the loss function used?</summary>

`2.2. Training`

</details>

Find and write down the loss function.

<details>
<summary>Reveal answer</summary>

![w05_loss.png](assets/w05_loss.png "w05_loss.png")

</details>

Before we dig into the components of the loss function, let's first understand what those $\lambda$s are.

<details>
<summary>What explanation do the authors give for the lambda terms?</summary>

- For most of the images, we'd only have a few cells that would be assigned with some target. All of the remaining cells will be considered "background".
- This essentially means we have a very unbalanced dataset.
- The loss would thus end-up favoring/teaching the network to be very good at predicting background vs object cases, and would not place enough emphasis on the quality of the object predictions since they contribute very loss values to the total loss.
- The authors solve this problem by:
  - increasing the contribution (i.e. **weight**) of the bouding box prediction errors via a term $\lambda_{coord}$;
  - decreasing the contribution (i.e. weight) of confidence predictions for boxes that don’t contain objects via a term $\lambda_{noobj}$.

![w05_loss_weights.png](assets/w05_loss_weights.png "w05_loss_weights.png")

Remember this technique! It is called **loss weighting** and is **one of the most effective ways of dealing with class imbalance**.

</details>

Analyze the first part of the loss function:

![w05_loss_p1.png](assets/w05_loss_p1.png "w05_loss_p1.png")

<details>
<summary>What does it mean? What added value does it have?</summary>

- We can consider this the localization loss.
- It measures the quality of the bounding box predictions.
- It's implemented as MSE (all components of the loss are MSE) between predicted offsets and predicted width and height:
- Additionally, an indicator function is used to keep only the cells $i$ for which a box $j$ holds the predictions for the target in $i$.

![w05_loss_p2.png](assets/w05_loss_p2.png "w05_loss_p2.png")

</details>

<details>
<summary>But here there's a catch - how are the two summations different?</summary>

The authors use the square root of the predicted and actual width and height.

</details>

<details>
<summary>Why is that?</summary>

We want to penalize small deviations more in smaller boxes than in larger ones.

![w05_loss_p3.png](assets/w05_loss_p3.png "w05_loss_p3.png")

The problem and the solution can be seen if we plot the width of the target box against the generated loss:

- blue line is $y=\left(\sqrt{w_{i}}-\sqrt{\left(w_{i} \pm 0.1\right)}\right)^{2}$
- red line is $y=\left(w_{i}-\left(w_{i} \pm 0.1\right)\right)^{2}$

![w05_loss_p3_gr.png](assets/w05_loss_p3_gr.png "w05_loss_p3_gr.png")

</details>

Analyze the second part of the loss function:

![w05_loss2_p1.png](assets/w05_loss2_p1.png "w05_loss2_p1.png")

<details>
<summary>What does it mean? What added value does it have?</summary>

- We can consider this the confidence loss.
- It measures how correct the model is at predicting the IoU of the chosen bouding box.
- We use two indicator functions this time:
  - the first checks how good the predicted IoU confidence is for boxes $j$ in which there is a cell $i$ in which there is a center of an object;
  - the second one checks how close the confidences of the boxes for which there is a not a cell with an object are to `0`.

</details>

Analyze the third part of the loss function:

![w05_loss3_p1.png](assets/w05_loss3_p1.png "w05_loss3_p1.png")

<details>
<summary>What does it mean? What added value does it have?</summary>

- We can consider this the classification loss.
- It measures how correct the model is at predicting the class of the object in cell $i$ that contains it.

</details>

In the beginning of the session we saw that an object detection model can return class specific confidence score for each predicted object:

![w05_od_output.png](assets/w05_od_output.png "w05_od_output.png")

<details>
<summary>How is this score obtained in test time (when we don't know the objects and boxes) in the case of YOLO?</summary>

The authors show this in the beginning of the paper - they multiply the probability coming from the model for how likely the box is to contain that class with the model's estimation for how well the predicted box fits that object.

</details>

## Evaluating object detection models

<details>
<summary>What is the main metric used by the authors?</summary>

Their main metric is the [mean Average Precision (mAP)](<https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision>):

![w05_metric.png](assets/w05_metric.png "w05_metric.png")

</details>

It is [related](<https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision>) to the area under the precision-recall curve in that evaluates how well predicted bounding boxes match ground-truth boxes at a fixed IoU threshold (typically IoU ≥ 0.5).

For each class, we compute **Average Precision (AP)** by combining precision
and recall into a single number (area under the precision–recall curve).
The final **mAP** is the mean of AP across all classes.

Let's derive this: for one class $c$:

- $N_{\text{gt}}$ = number of ground-truth objects
- $N$ = number of predicted detections (sorted by confidence, descending)

Each prediction is matched using IoU ≥ 0.5:

$$
\text{TP}_k =
\begin{cases}
1 & \text{if prediction } k \text{ matches an unused ground-truth box} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\text{FP}_k = 1 - \text{TP}_k
$$

We obtain the cumulative counts:

$$
TP_k^{\text{cum}} = \sum_{i=1}^{k} \text{TP}_i
$$

$$
FP_k^{\text{cum}} = \sum_{i=1}^{k} \text{FP}_i
$$

Precision and recall at step $k$:

$$
P_k = \frac{TP_k^{\text{cum}}}{TP_k^{\text{cum}} + FP_k^{\text{cum}}}
$$

$$
R_k = \frac{TP_k^{\text{cum}}}{N_{\text{gt}}}
$$

with $R_0 = 0$.

Recall increases in discrete steps:

$$
\Delta R_k = R_k - R_{k-1}
$$

Since the precision–recall curve is not necessarily monotonic, we define:

$$
P_k^{\text{interp}} = \max_{j \ge k} P_j
$$

This ensures a monotonically decreasing precision curve.

AP is then the discrete area under the precision–recall curve:

$$
AP_c = \sum_{k=1}^{N} (R_k - R_{k-1}) \, P_k^{\text{interp}}
$$

For $C$ classes:

$$
mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c
$$

This corresponds to the modern VOC (2010+) discrete integration method, but note that other datasets (ex. [COCO](https://docs.ultralytics.com/datasets/detect/coco/)) may use other methods.

How to use in PyTorch:

```python
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

preds = [{
    "boxes": torch.tensor([[10, 15, 100, 120]], dtype=torch.float32),  # xyxy
    "scores": torch.tensor([0.92], dtype=torch.float32),
    "labels": torch.tensor([0], dtype=torch.int64),
}]
targets = [{
    "boxes": torch.tensor([[12, 18, 98, 118]], dtype=torch.float32),   # xyxy
    "labels": torch.tensor([0], dtype=torch.int64),
}]

metric.update(preds, targets)
out = metric.compute()

print(out["map"])
```
