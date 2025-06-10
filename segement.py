import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
class SAMSegmentation:
    def __init__(self, image_path="rgb_bn.png", model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
        """
        Initialize SAM segmentation model
        :param image_path: Path to input image
        :param model_type: SAM model type (vit_b, vit_l, vit_h)
        :param checkpoint_path: Path to SAM checkpoint file
        """
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        
        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        # Load image
        self.image_path = image_path
        self.image = np.array(Image.open(image_path).convert("RGB"))
        self.masks = None

    def process_image(self):
        """Generate segmentation masks for the loaded image"""
        self.masks = self.mask_generator.generate(self.image)
        return self.masks

    @staticmethod
    def get_boundary_coordinates(mask):
        """Convert segmentation mask to boundary coordinates"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        simplified_contours = []
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(approx)
        
        boundary_coords = []
        for cnt in simplified_contours:
            coords = cnt.squeeze().tolist()
            if len(coords) > 2:
                boundary_coords.append(coords)
        
        return boundary_coords

    def analyze_masks(self):
        """Process all masks and return boundaries with metadata"""
        if self.masks is None:
            raise ValueError("Run process_image() first")
            
        results = []
        for mask_data in self.masks:
            boundary = {
                'coordinates': self.get_boundary_coordinates(mask_data['segmentation']),
                'area': mask_data['area'],
                'predicted_iou': mask_data['predicted_iou']
            }
            results.append(boundary)
        return results

    def visualize_masks(self):
        """Visualize segmentation results"""
        if self.masks is None:
            raise ValueError("Run process_image() first")
            
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        for mask in self.masks:
            m = mask["segmentation"]
            color = np.concatenate([np.random.random(3), [0.35]])
            plt.imshow(m, alpha=0.5, cmap='gray', 
                      extent=[0, self.image.shape[1], self.image.shape[0], 0])
        plt.axis('off')
        plt.show()

    def visualize_boundaries(self):
        """Visualize segmentation boundaries"""
        if self.masks is None:
            raise ValueError("Run process_image() first")
            
        image_with_contours = self.image.copy()
        for mask_data in self.masks:
            coords = self.get_boundary_coordinates(mask_data["segmentation"])
            for contour in coords:
                pts = np.array(contour, dtype=np.int32)
                cv2.polylines(image_with_contours, [pts], 
                             isClosed=True, color=(0,255,0), thickness=2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_contours)
        plt.axis('off')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize processor
    segmenter = SAMSegmentation(image_path="rgb_bn.png")
    
    # Process image
    segmenter.process_image()
    
    # Get analysis results
    results = segmenter.analyze_masks()
    
    print("Number of masks:", results)
    # Print first mask's first 5 coordinates
    print("First mask boundary coordinates (first 5 points):")
    print(results[0]['coordinates'][0][:5])
    
    # Visualize results
    segmenter.visualize_masks()
    # segmenter.visualize_boundaries()