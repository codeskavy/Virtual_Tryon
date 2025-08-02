import cv2
import numpy as np
import torch
import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# Configuration paths
GROUNDING_DINO_CONFIG_PATH = './weights/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = './weights/groundingdino_swint_ogc.pth'
SAM_CHECKPOINT_PATH = './weights/sam_vit_h_4b8939.pth'

class ClothingDetector:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gd_model = None
        self.sam_model = None
        self.mask_predictor = None
    
    def load_models(self):
        """Load GroundingDINO and SAM models"""
        self.gd_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
        )
        
        self.sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=self.device)
        self.mask_predictor = SamPredictor(self.sam_model)
    
    def detect_and_segment(self, image, classes=['tshirt'], box_threshold=0.35, text_threshold=0.25):
        """Main detection and segmentation function"""
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image_bgr, (1024, 1024))
        image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        detections = self.gd_model.predict_with_classes(
            image=image_rgb_resized,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Segment using SAM
        segmented_masks = []
        detected_boxes = detections.xyxy
        
        self.mask_predictor.set_image(image_rgb_resized)
        
        for box in detected_boxes:
            box_np = np.array(box)
            masks, scores, logits = self.mask_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=False
            )
            segmented_masks.append(masks)
        
        if segmented_masks:
            combined_mask = self.combine_masks(segmented_masks, image_resized)
            return combined_mask, detections
        
        return None, detections
    
    def combine_masks(self, segmented_masks, original_image):
        """Combine multiple segmentation masks"""
        combined_mask = segmented_masks[0][0].astype(np.uint8) * 255
        
        for i in range(1, len(segmented_masks)):
            mask = segmented_masks[i][0].astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        mask_3d = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        segmented_image = cv2.bitwise_and(original_image, mask_3d)
        segmented_image[np.where((segmented_image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        
        return segmented_image

detector = ClothingDetector()

def detect_and_segment_clothing(image, classes=['tshirt'], box_threshold=0.35, text_threshold=0.25):
    if detector.gd_model is None:
        detector.load_models()
    return detector.detect_and_segment(image, classes, box_threshold, text_threshold)