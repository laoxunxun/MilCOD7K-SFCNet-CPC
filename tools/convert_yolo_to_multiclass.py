# -*- coding: utf-8 -*-
"""
Correctly convert a YOLO-seg dataset to multi-class segmentation format
- class 0: background
- class 1: camouflage soldier (original YOLO class 0)
- class 2: military vehicle (original YOLO class 1)
- class 3: tank (original YOLO class 2)
- class 4: defense fortification (original YOLO class 3)
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


class YOLOToMultiClassConverterFixed:
    """Convert a YOLO-segmentation dataset to multi-class segmentation (with a background class)"""

    # class-name mapping
    CLASS_NAMES = {
        0: 'background',               # background
        1: 'camouflage_soldier',      # camouflage soldier (original YOLO class 0)
        2: 'military_vehicle',         # military vehicle (original YOLO class 1)
        3: 'tank',                     # tank (original YOLO class 2)
        4: 'fortification'     # defense fortification (original YOLO class 3)
    }

    def __init__(self, yolo_img_dir, yolo_label_dir, output_base_dir, num_classes=5):
        # num_classes = 5 (background + 4 object classes)
        self.yolo_img_dir = yolo_img_dir
        self.yolo_label_dir = yolo_label_dir
        self.output_base_dir = output_base_dir
        # actual number of classes = 5 (background + 4 foreground classes)
        self.num_classes = num_classes
        # number of YOLO classes (excluding background)
        self.num_yolo_classes = 4

    def parse_yolo_segmentation(self, label_path, img_width, img_height):
        """
        Parse YOLO segmentation labels
        Format: class_id x1 y1 x2 y2 ... xn yn (normalized coordinates)
        Return: list of {class_id, coords}
        """
        masks = []

        if not os.path.exists(label_path):
            return masks

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:  # need at least a class ID + 1 point
                    continue

                yolo_class_id = int(parts[0])
                if yolo_class_id >= self.num_yolo_classes:
                    continue

                # map YOLO class IDs to the new class IDs (original class 0 -> new class 1, etc.)
                new_class_id = yolo_class_id + 1

                # extract the polygon coordinate points
                coords = []
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        x = float(parts[i]) * img_width
                        y = float(parts[i + 1]) * img_height
                        coords.append([x, y])

                if len(coords) >= 3:
                    masks.append({
                        'class_id': new_class_id,  # use the new class ID (1-4)
                        'coords': np.array(coords, dtype=np.int32)
                    })

        return masks

    def masks_to_one_hot(self, masks, img_height, img_width):
        """
        Convert multiple masks to one-hot encoding
        Return: an H x W x C numpy array, uint8
        C = num_classes (5 = background + 4 objects)
        """
        # build the one-hot encoding; initially the background class (class 0) is all 1
        one_hot = np.zeros((img_height, img_width, self.num_classes), dtype=np.uint8)
        one_hot[:, :, 0] = 1  # background class initialized to 1

        # draw the foreground objects, covering the background
        for mask_info in masks:
            class_id = mask_info['class_id']
            coords = mask_info['coords']

            # create the polygon mask
            contour = coords.reshape((-1, 1, 2))
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)

            # add to the corresponding class channel
            one_hot[:, :, class_id] = np.maximum(one_hot[:, :, class_id], mask)
            # clear the background label at that position
            one_hot[:, :, 0] = np.where(mask > 0, 0, one_hot[:, :, 0])

        return one_hot

    def generate_edge(self, one_hot_mask):
        """Generate edge maps (edges of all foreground classes)"""
        # get the union of all foreground classes
        foreground = one_hot_mask[:, :, 1:].max(axis=2)  # H x W

        # detect edges with Canny
        edges = cv2.Canny(foreground * 255, 100, 200)

        # dilate the edges to make them more visible
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def convert_split(self, split='train'):
        """Convert one split of the dataset"""
        print(f"\n{'='*60}")
        print(f"Converting {split} dataset")
        print(f"{'='*60}")

        # output directory
        img_out_dir = os.path.join(self.output_base_dir, split, 'Imgs')
        gt_out_dir = os.path.join(self.output_base_dir, split, 'GT')
        edge_out_dir = os.path.join(self.output_base_dir, split, 'Edge')

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(gt_out_dir, exist_ok=True)
        os.makedirs(edge_out_dir, exist_ok=True)

        # get the image list
        img_files = sorted([f for f in os.listdir(self.yolo_img_dir)
                           if f.endswith(('.jpg', '.png', '.jpeg'))])

        if len(img_files) == 0:
            print(f"Warning: no images found in {self.yolo_img_dir}")
            return

        print(f"Found {len(img_files)} images")

        stats = {i: 0 for i in range(5)}  # 5 classes: 0(background) + 4 objects

        for img_file in tqdm(img_files, desc=f"Converting {split}"):
            # read the image
            img_path = os.path.join(self.yolo_img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_height, img_width = img.shape[:2]

            # the corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.yolo_label_dir, label_file)

            # parse the YOLO labels
            masks = self.parse_yolo_segmentation(label_path, img_width, img_height)

            if not masks:
                # for unlabeled images, create a background-only mask
                one_hot = np.zeros((img_height, img_width, self.num_classes), dtype=np.uint8)
                one_hot[:, :, 0] = 1  # all background
                stats[0] += 1  # count as one background sample
            else:
                # convert to a one-hot mask
                one_hot = self.masks_to_one_hot(masks, img_height, img_width)

                # count the classes
                for mask_info in masks:
                    stats[mask_info['class_id']] += 1

            # save the image
            img_save_path = os.path.join(img_out_dir, img_file)
            if not img_save_path.endswith('.jpg'):
                img_save_path = os.path.splitext(img_save_path)[0] + '.jpg'
            cv2.imwrite(img_save_path, img)

            # save the GT mask (one-hot)
            gt_save_path = os.path.join(gt_out_dir,
                                       os.path.splitext(img_file)[0] + '.npy')
            np.save(gt_save_path, one_hot)

            # generate and save the edge maps
            edges = self.generate_edge(one_hot)
            edge_save_path = os.path.join(edge_out_dir,
                                         os.path.splitext(img_file)[0] + '.png')
            cv2.imwrite(edge_save_path, edges)

        # print the statistics
        print(f"\nClass statistics:")
        for class_id, count in stats.items():
            print(f"  class {class_id} ({self.CLASS_NAMES[class_id]}): {count} instances")
        print(f"\nSave location:")
        print(f"  images: {img_out_dir}")
        print(f"  GT masks: {gt_out_dir}")
        print(f"  edges: {edge_out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_img_dir', type=str,
                       default='G:/work1/data/datasets/images/train',
                       help='YOLO-format image directory')
    parser.add_argument('--yolo_label_dir', type=str,
                       default='G:/work1/data/datasets/labels/train',
                       help='YOLO-format label directory')
    parser.add_argument('--output_dir', type=str,
                       default='./Dataset_multiclass_5class/',
                       help='output directory')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='number of classes (including background)')

    args = parser.parse_args()

    print("="*60)
    print("Convert a YOLO-segmentation dataset to multi-class segmentation format (with a background class)")
    print("="*60)
    print(f"\nClass definitions:")
    converter = YOLOToMultiClassConverterFixed('', '', '', args.num_classes)
    for class_id, name in converter.CLASS_NAMES.items():
        print(f"  class {class_id}: {name}")

    # convert the training set
    train_converter = YOLOToMultiClassConverterFixed(
        args.yolo_img_dir,
        args.yolo_label_dir,
        args.output_dir,
        args.num_classes
    )
    train_converter.convert_split('train')

    # convert the validation set
    val_img_dir = args.yolo_img_dir.replace('/train/', '/val/')
    val_label_dir = args.yolo_label_dir.replace('/train/', '/val/')

    if os.path.exists(val_img_dir):
        val_converter = YOLOToMultiClassConverterFixed(
            val_img_dir,
            val_label_dir,
            args.output_dir,
            args.num_classes
        )
        val_converter.convert_split('val')
    else:
        print(f"\nWarning: validation directory does not exist: {val_img_dir}")

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)


if __name__ == '__main__':
    main()
