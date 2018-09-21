import torch
from torch.autograd import Variable
import cv2
import imageio

from data import BaseTransform, VOC_CLASSES as label_map
from ssd import build_ssd


# input the frame, the neural net, and the transformer for nn input
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    # RBG --> GRB; the model was trained on GRB, and we get RBG
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    # network only accepts "batches", so we fake a dimension
    x = Variable(x.unsqueeze(0))
    y = net(x)

    detections = y.data
    # shape of detection: [batch, number of classes, number of class
    # occurrences, (score, x0, y0, x1, y1)]
    scale = torch.Tensor([width, height, width, height])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] > 0.2:
            point = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(
                frame,
                pt1=(int(point[0]), int(point[1])),
                pt2=(int(point[2]), int(point[3])),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.putText(
                frame,
                text=label_map[i - 1],
                org=(int(point[0]), int(point[1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            j += 1

    return frame


# create the ssd neural network (the pretrained one)
net = build_ssd('test')
net.load_state_dict(
    torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc: storage)
)

# create image transformation; magic mean numbers
transform = BaseTransform(net.size, (104 / 256, 117 / 256, 123 / 256))

# load video and apply detection
reader = imageio.get_reader('funny_dog.mp4')
video_meta = reader.get_meta_data()
fps = video_meta['fps']
writer = imageio.get_writer('output.mp4', fps=fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print('Processed {}/{}'.format(i + 1, video_meta['nframes']))
writer.close()
