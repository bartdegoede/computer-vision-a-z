# Download PyTorch weights
Put `ssd300_mAP_77.42_v2.pth` in this folder. Download it [here](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth), more information [here](https://github.com/amdegroot/ssd.pytorch/#use-a-pre-trained-ssd-network-for-detection).

## PyTorch 0.4.0 breaks the course materials
If you follow along with the [Computer Vision A-Z course on Udemy](https://www.udemy.com/computer-vision-a-z/), the API changes in [PyTorch 0.4.0](https://github.com/pytorch/pytorch/releases/tag/v0.4.0) break some of the magic happening in the provided SSD code. I managed to get some detection going on by lowering the detection score threshold, but that didn't really let me recognize the dog:

![dogs-are-people](object_detection/dogs-are-people.png)
