"""
SFCNet multi-class segmentation inference script
- support both image and video inference
- overlay class annotations and segmentation masks on the original image
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm
import argparse

sys.path.append('./models')
from models.NetMultiClass import NetMultiClass


# class config
CLASS_NAMES = [
    'background',
    'camouflage_soldier',
    'military_vehicle',
    'tank',
    'fortification'
]

# visualization color for each class (BGR)
CLASS_COLORS = [
    [0, 0, 0],             # background - not shown
    [0, 0, 255],           # camouflage_soldier - red
    [0, 165, 255],         # military_vehicle - orange
    [255, 0, 0],           # tank - blue
    [0, 255, 255],         # fortification - yellow
]

# label text color (BGR)
LABEL_COLORS = [
    (0, 0, 0),
    (0, 0, 255),
    (0, 165, 255),
    (255, 0, 0),
    (0, 255, 255),
]


def load_model(weight_path, num_classes, gpu_id='0'):
    """Load the multi-class model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Loading model weights: {weight_path}")

    model = NetMultiClass(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=False))
    model = model.to(device)
    model.eval()

    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")
    return model, device


def get_transform(test_size=384):
    """Get image preprocessing"""
    return transforms.Compose([
        transforms.Resize((test_size, test_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict(model, device, image_bgr, transform):
    """Run a multi-class segmentation prediction on a single image"""
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        pred_labels = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # get the per-class probability maps
    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    return pred_labels, probs


def create_colored_mask(pred_labels, num_classes):
    """Generate a colored segmentation mask"""
    h, w = pred_labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(1, num_classes):  # skip background
        color = CLASS_COLORS[cls_id]
        mask[pred_labels == cls_id] = color

    return mask


def create_overlay(image_bgr, pred_labels, probs, num_classes, alpha=0.45):
    """
    Overlay a semi-transparent segmentation mask on the original image, with class labels and contours
    """
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()

    for cls_id in range(1, num_classes):
        cls_mask = (pred_labels == cls_id).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue

        # semi-transparent mask overlay
        color = np.array(CLASS_COLORS[cls_id], dtype=np.uint8)
        color_layer = np.full_like(overlay, color)
        overlay[cls_mask > 0] = cv2.addWeighted(
            overlay[cls_mask > 0], 1 - alpha,
            color_layer[cls_mask > 0], alpha, 0
        ).astype(np.uint8)

        # draw the contours
        contours, _ = cv2.findContours(
            cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), thickness=2)

        # annotate the class name at the center of each connected region
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # region too small to annotate with text
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # get the average confidence of the region
            cls_prob = probs[cls_id]
            avg_conf = cls_prob[cls_mask > 0].mean()

            label = f"{CLASS_NAMES[cls_id]} {avg_conf:.0%}"

            # compute the font size (adaptive to the region)
            font_scale = max(0.4, min(0.7, area / 50000))
            thickness = max(1, int(font_scale * 2))

            # get the text size
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # draw the text background
            label_x = cx - tw // 2
            label_y = cy - th // 2
            bg_x1 = max(0, label_x - 3)
            bg_y1 = max(0, label_y - 3)
            bg_x2 = min(w, label_x + tw + 3)
            bg_y2 = min(h, label_y + th + baseline + 3)

            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                          (0, 0, 0), -1)

            # draw the text
            cv2.putText(overlay, label, (label_x, label_y + th),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def draw_legend(canvas, num_classes, start_x=10, start_y=10):
    """Draw the class legend on the image"""
    font_scale = 0.6
    thickness = 1
    line_height = 25
    padding = 5
    bg_width = 220
    bg_height = num_classes * line_height + padding * 2

    # draw a semi-transparent background
    overlay_bg = canvas.copy()
    cv2.rectangle(overlay_bg, (start_x, start_y),
                  (start_x + bg_width, start_y + bg_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay_bg, 0.6, canvas, 0.4, 0, canvas)

    for i in range(1, num_classes):
        y = start_y + padding + (i - 1) * line_height + line_height // 2

        # color block
        cv2.rectangle(canvas, (start_x + padding, y - 6),
                      (start_x + padding + 18, y + 6),
                      tuple(CLASS_COLORS[i]), -1)

        # text
        cv2.putText(canvas, CLASS_NAMES[i],
                    (start_x + padding + 25, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas


def process_image(model, device, image_path, output_path, transform, num_classes):
    """Process a single image"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  failed to read image: {image_path}")
        return

    pred_labels, probs = predict(model, device, image_bgr, transform)

    # generate the overlay image
    result = create_overlay(image_bgr, pred_labels, probs, num_classes, alpha=0.45)

    # generate the colored mask
    colored_mask = create_colored_mask(pred_labels, num_classes)

    # add the legend
    result = draw_legend(result, num_classes)

    # concat: original | overlay | colored mask
    combined = np.hstack([image_bgr, result, colored_mask])

    cv2.imwrite(output_path, combined)
    print(f"  save: {output_path}")

    # also save a separate annotation-result image
    single_output = output_path.replace('.', '_annotated.', 1)
    cv2.imwrite(single_output, result)

    return True


def process_video(model, device, video_path, output_path, transform, num_classes):
    """Process a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  failed to open video: {video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  resolution: {width}x{height}, fps: {fps}, total frames: {total_frames}")

    # output width = original + annotation + mask, with a legend bar on top
    legend_h = 40
    out_w = width * 3
    out_h = height + legend_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # legend bar (drawn only once)
    legend_bar = np.zeros((legend_h, out_w, 3), dtype=np.uint8)
    cv2.putText(legend_bar, "SFCNet Multi-Class Segmentation",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i in range(1, num_classes):
        x = 400 + (i - 1) * 180
        cv2.rectangle(legend_bar, (x, 12), (x + 18, 28),
                      tuple(CLASS_COLORS[i]), -1)
        cv2.putText(legend_bar, CLASS_NAMES[i], (x + 23, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="  Processing frames")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pred_labels, probs = predict(model, device, frame, transform)
            annotated = create_overlay(frame, pred_labels, probs, num_classes, alpha=0.45)
            colored_mask = create_colored_mask(pred_labels, num_classes)

            combined = np.hstack([frame, annotated, colored_mask])
            full_frame = np.vstack([legend_bar, combined])

            out.write(full_frame)
            pbar.update(1)
        pbar.close()

    cap.release()
    out.release()
    print(f"  save video: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='SFCNet multi-class segmentation inference')
    parser.add_argument('--weight', type=str,
                        default='./cpts_multiclass_5class/Net_multi_best_iou.pth',
                        help='model weight path (best IoU)')
    parser.add_argument('--input_dir', type=str,
                        default='./test_lettle_data',
                        help='input directory (images and videos)')
    parser.add_argument('--output_dir', type=str,
                        default='./inference_results',
                        help='output directory')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--test_size', type=int, default=384)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()

    print("=" * 60)
    print("SFCNet multi-class segmentation inference")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model weights: {args.weight}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Classes: {', '.join([f'{i}={n}' for i, n in enumerate(CLASS_NAMES[:args.num_classes])])}")
    print("=" * 60)

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load the model
    model, device = load_model(args.weight, args.num_classes, args.gpu_id)
    transform = get_transform(args.test_size)

    # scan the input files
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    files = sorted(os.listdir(args.input_dir))
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_exts]
    videos = [f for f in files if os.path.splitext(f)[1].lower() in video_exts]

    print(f"\nFound {len(images)} images, {len(videos)} videos\n")

    # process images
    if images:
        print("--- Processing images ---")
        for fname in images:
            print(f"  [{fname}]")
            input_path = os.path.join(args.input_dir, fname)
            output_path = os.path.join(args.output_dir, fname)
            process_image(model, device, input_path, output_path, transform, args.num_classes)
        print()

    # process video
    if videos:
        print("--- Processing video ---")
        for fname in videos:
            print(f"  [{fname}]")
            input_path = os.path.join(args.input_dir, fname)
            name, ext = os.path.splitext(fname)
            output_path = os.path.join(args.output_dir, f"{name}_result{ext}")
            process_video(model, device, input_path, output_path, transform, args.num_classes)
        print()

    print("=" * 60)
    print(f"Inference complete! Results saved in: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
