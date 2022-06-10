import re
import glob
import json
import numpy
import torch
import cv2

from torchvision import transforms, models
import matplotlib.pyplot as plt


IMAGE_SIZE = (256, 192)

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_3rd_point(a, b):
    direct = a - b
    return b + numpy.array([-direct[1], direct[0]], dtype=numpy.float32)

def get_person_detection_boxes(model, img, threshold):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = numpy.zeros((2), dtype=numpy.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = numpy.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=numpy.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def get_dir(src_point, rot_rad):
    sn, cs = numpy.sin(rot_rad), numpy.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(
        center, scale, rot, output_size,
        shift=numpy.array([0, 0], dtype=numpy.float32), inv=0
):
    if not isinstance(scale, numpy.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = numpy.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = numpy.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = numpy.array([0, dst_w * -0.5], numpy.float32)

    src = numpy.zeros((3, 2), dtype=numpy.float32)
    dst = numpy.zeros((3, 2), dtype=numpy.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = numpy.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(numpy.float32(dst), numpy.float32(src))
    else:
        trans = cv2.getAffineTransform(numpy.float32(src), numpy.float32(dst))

    return trans

def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = numpy.max(hm[i, j])
            dr = numpy.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i, j] = dr[border: -border, border: -border].copy()
            hm[i, j] *= origin_max / numpy.max(hm[i, j])
    return hm

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, numpy.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = numpy.argmax(heatmaps_reshaped, 2)
    maxvals = numpy.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = numpy.tile(idx, (1, 1, 2)).astype(numpy.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = numpy.floor((preds[:, :, 1]) / width)

    pred_mask = numpy.tile(numpy.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(numpy.float32)

    preds *= pred_mask
    return preds, maxvals

def affine_transform(pt, t):
    new_pt = numpy.array([pt[0], pt[1], 1.]).T
    new_pt = numpy.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = numpy.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1]
                      + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = numpy.matrix([[dx], [dy]])
        hessian = numpy.matrix([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = numpy.squeeze(numpy.array(offset.T), axis=0)
            coord += offset
    return coord

def get_final_preds(hm, center, scale, transform_back=True, test_blur_kernel=3):
    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]

    # post-processing
    hm = gaussian_blur(hm, test_blur_kernel)
    hm = numpy.maximum(hm, 1e-10)
    hm = numpy.log(hm)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            coords[n, p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    if transform_back:
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

    return preds, maxvals

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0
    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            output.clone().cpu().numpy(),
            numpy.asarray([center]),
            numpy.asarray([scale]))

        return preds

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


box_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
box_model.to(CTX)
box_model.eval()

tpr = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True)

def process_image(image_path):
    print(f"Processing image {image_path}")
    input = []
    img_rgb = cv2.imread(image_path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_tensor = torch.from_numpy(img_rgb / 255.).permute(2, 0, 1).float().to(CTX)
    input.append(img_tensor)

    pred_boxes = get_person_detection_boxes(box_model, input, 0.7)
    keypoints = []

    if len(pred_boxes) >= 1:
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, IMAGE_SIZE[0], IMAGE_SIZE[1])
            image_pose = img_rgb.copy()
            pose_preds = get_pose_estimation_prediction(tpr, image_pose, center, scale)
            if len(pose_preds) >= 1:
                for kpt in pose_preds:
                    keypoints.append(kpt.tolist())
                    draw_pose(kpt, img_bgr)

    return img_bgr, keypoints

def save_metadata(image_folder):
    image_re = re.compile(r"Images/.*?/(.*)\.")
    for image_path in glob.glob(f"Images/{image_folder}/*"):
        img, kpts = process_image(image_path)

        f = plt.figure()
        plt.imshow(img)

        image_name = image_re.search(image_path).group(1)
        pose_image_path = f"PoseImages/{image_folder}/{image_name}.png"
        pose_json_path = f"Poses/{image_folder}/{image_name}.json"
        f.savefig(pose_image_path)
        plt.cla()

        with open(pose_json_path, "w") as fp:
            json.dump({ "keypoints": kpts, "tag": image_folder }, fp, indent=4)

def main():
    save_metadata("Roubo")
    save_metadata("NaoRoubo")

if __name__ == "__main__":
    main()