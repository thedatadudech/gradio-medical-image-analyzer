#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fat Segmentation Module for Medical Image Analyzer
Implements subcutaneous and visceral fat detection using HU values
Adapted for multi-species veterinary and human medical imaging
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from typing import Tuple, Dict, Any, Optional
import cv2


class FatSegmentationEngine:
    """
    Engine for detecting and segmenting fat tissue in medical CT images
    Supports both human and veterinary applications
    """
    
    # HU ranges for different tissues
    FAT_HU_RANGE = (-190, -30)  # Fat tissue Hounsfield Units
    MUSCLE_HU_RANGE = (30, 80)   # Muscle tissue for body outline
    
    def __init__(self):
        self.fat_mask = None
        self.subcutaneous_mask = None
        self.visceral_mask = None
        self.body_outline = None
        
    def segment_fat_tissue(self, hu_array: np.ndarray, species: str = "human") -> Dict[str, Any]:
        """
        Segment fat tissue into subcutaneous and visceral components
        
        Args:
            hu_array: 2D array of Hounsfield Unit values
            species: Species type for specific adjustments
            
        Returns:
            Dictionary containing segmentation results and statistics
        """
        # Step 1: Detect all fat tissue based on HU values
        self.fat_mask = self._detect_fat_tissue(hu_array)
        
        # Step 2: Detect body outline for subcutaneous/visceral separation
        self.body_outline = self._detect_body_outline(hu_array)
        
        # Step 3: Separate subcutaneous and visceral fat
        self.subcutaneous_mask, self.visceral_mask = self._separate_fat_types(
            self.fat_mask, self.body_outline, species
        )
        
        # Step 4: Calculate statistics
        stats = self._calculate_fat_statistics(hu_array)
        
        # Step 5: Add clinical assessment
        assessment = self.assess_obesity_risk(stats, species)
        
        return {
            'fat_mask': self.fat_mask,
            'subcutaneous_mask': self.subcutaneous_mask,
            'visceral_mask': self.visceral_mask,
            'body_outline': self.body_outline,
            'statistics': stats,
            'assessment': assessment,
            'overlay_colors': self._generate_overlay_colors()
        }
    
    def _detect_fat_tissue(self, hu_array: np.ndarray) -> np.ndarray:
        """Detect fat tissue based on HU range"""
        fat_mask = (hu_array >= self.FAT_HU_RANGE[0]) & (hu_array <= self.FAT_HU_RANGE[1])
        
        # Clean up noise with morphological operations
        fat_mask = morphology.binary_opening(fat_mask, morphology.disk(2))
        fat_mask = morphology.binary_closing(fat_mask, morphology.disk(3))
        
        return fat_mask.astype(np.uint8)
    
    def _detect_body_outline(self, hu_array: np.ndarray) -> np.ndarray:
        """
        Detect body outline to separate subcutaneous from visceral fat
        Uses muscle tissue and air/tissue boundaries
        """
        # Threshold for air/tissue boundary (around -500 HU)
        tissue_mask = hu_array > -500
        
        # Fill holes and smooth the outline
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)
        tissue_mask = morphology.binary_closing(tissue_mask, morphology.disk(5))
        
        # Find the largest connected component (main body)
        labeled = measure.label(tissue_mask)
        props = measure.regionprops(labeled)
        
        if props:
            largest_region = max(props, key=lambda x: x.area)
            body_mask = (labeled == largest_region.label)
            
            # Get the outline/contour
            outline = morphology.binary_erosion(body_mask, morphology.disk(10))
            outline = body_mask & ~outline
            
            return outline.astype(np.uint8)
        
        return np.zeros_like(hu_array, dtype=np.uint8)
    
    def _separate_fat_types(self, fat_mask: np.ndarray, body_outline: np.ndarray, 
                           species: str = "human") -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate subcutaneous and visceral fat based on body outline
        """
        # Distance transform from body outline
        distance_from_outline = ndimage.distance_transform_edt(~body_outline.astype(bool))
        
        # Adjust threshold based on species
        distance_thresholds = {
            "human": 25,
            "dog": 20,
            "cat": 15,
            "horse": 30,
            "cattle": 35
        }
        
        threshold = distance_thresholds.get(species.lower(), 20)
        
        # Subcutaneous fat: fat tissue close to body surface
        subcutaneous_mask = fat_mask & (distance_from_outline <= threshold)
        
        # Visceral fat: remaining fat tissue inside the body
        visceral_mask = fat_mask & ~subcutaneous_mask
        
        # Additional filtering for visceral fat (inside body cavity)
        body_center = self._find_body_center(body_outline)
        visceral_mask = self._filter_visceral_fat(visceral_mask, body_center, species)
        
        return subcutaneous_mask.astype(np.uint8), visceral_mask.astype(np.uint8)
    
    def _find_body_center(self, body_outline: np.ndarray) -> Tuple[int, int]:
        """Find the center of the body for visceral fat filtering"""
        y_coords, x_coords = np.where(body_outline > 0)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            return (center_y, center_x)
        return (body_outline.shape[0] // 2, body_outline.shape[1] // 2)
    
    def _filter_visceral_fat(self, visceral_mask: np.ndarray, body_center: Tuple[int, int],
                            species: str = "human") -> np.ndarray:
        """
        Filter visceral fat to ensure it's inside the body cavity
        """
        # Create a mask for the central body region
        center_y, center_x = body_center
        h, w = visceral_mask.shape
        
        # Adjust ellipse size based on species
        ellipse_factors = {
            "human": (0.35, 0.35),
            "dog": (0.3, 0.3),
            "cat": (0.25, 0.25),
            "horse": (0.4, 0.35),
            "cattle": (0.4, 0.4)
        }
        
        x_factor, y_factor = ellipse_factors.get(species.lower(), (0.3, 0.3))
        
        # Create elliptical region around body center for visceral fat
        y, x = np.ogrid[:h, :w]
        ellipse_mask = ((x - center_x) / (w * x_factor))**2 + ((y - center_y) / (h * y_factor))**2 <= 1
        
        # Keep only visceral fat within the central body region
        filtered_visceral = visceral_mask & ellipse_mask
        
        return filtered_visceral.astype(np.uint8)
    
    def _calculate_fat_statistics(self, hu_array: np.ndarray) -> Dict[str, float]:
        """Calculate fat tissue statistics"""
        total_pixels = hu_array.size
        fat_pixels = np.sum(self.fat_mask > 0)
        subcutaneous_pixels = np.sum(self.subcutaneous_mask > 0)
        visceral_pixels = np.sum(self.visceral_mask > 0)
        
        # Calculate percentages
        fat_percentage = (fat_pixels / total_pixels) * 100
        subcutaneous_percentage = (subcutaneous_pixels / total_pixels) * 100
        visceral_percentage = (visceral_pixels / total_pixels) * 100
        
        # Calculate ratio
        visceral_subcutaneous_ratio = (
            visceral_pixels / subcutaneous_pixels 
            if subcutaneous_pixels > 0 else 0
        )
        
        # Calculate mean HU values for each tissue type
        fat_hu_mean = np.mean(hu_array[self.fat_mask > 0]) if fat_pixels > 0 else 0
        
        return {
            'total_fat_percentage': round(fat_percentage, 2),
            'subcutaneous_fat_percentage': round(subcutaneous_percentage, 2),
            'visceral_fat_percentage': round(visceral_percentage, 2),
            'visceral_subcutaneous_ratio': round(visceral_subcutaneous_ratio, 3),
            'total_fat_pixels': int(fat_pixels),
            'subcutaneous_fat_pixels': int(subcutaneous_pixels),
            'visceral_fat_pixels': int(visceral_pixels),
            'fat_mean_hu': round(float(fat_hu_mean), 1)
        }
    
    def _generate_overlay_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate color overlays for visualization"""
        return {
            'subcutaneous': (255, 255, 0, 128),  # Yellow with transparency
            'visceral': (255, 0, 0, 128),        # Red with transparency
            'body_outline': (0, 255, 0, 255)     # Green outline
        }
    
    def create_overlay_image(self, base_image: np.ndarray) -> np.ndarray:
        """
        Create overlay image with fat segmentation highlighted
        
        Args:
            base_image: Original grayscale DICOM image (0-255)
            
        Returns:
            RGB image with fat overlays
        """
        # Convert to RGB
        overlay_img = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        
        colors = self._generate_overlay_colors()
        
        # Apply subcutaneous fat overlay (yellow)
        if self.subcutaneous_mask is not None:
            yellow_overlay = np.zeros_like(overlay_img)
            yellow_overlay[self.subcutaneous_mask > 0] = colors['subcutaneous'][:3]
            overlay_img = cv2.addWeighted(overlay_img, 0.7, yellow_overlay, 0.3, 0)
        
        # Apply visceral fat overlay (red)
        if self.visceral_mask is not None:
            red_overlay = np.zeros_like(overlay_img)
            red_overlay[self.visceral_mask > 0] = colors['visceral'][:3]
            overlay_img = cv2.addWeighted(overlay_img, 0.7, red_overlay, 0.3, 0)
        
        # Add body outline (green)
        if self.body_outline is not None:
            overlay_img[self.body_outline > 0] = colors['body_outline'][:3]
        
        return overlay_img
    
    def assess_obesity_risk(self, fat_stats: Dict[str, float], species: str = "human") -> Dict[str, Any]:
        """
        Assess obesity based on fat percentage and species-specific thresholds
        
        Args:
            fat_stats: Fat statistics from segmentation
            species: Species type (human, dog, cat, etc.)
            
        Returns:
            Obesity assessment with recommendations
        """
        # Species-specific fat percentage thresholds
        thresholds = {
            'human': {
                'normal': (10, 20),
                'overweight': (20, 30),
                'obese': (30, float('inf'))
            },
            'dog': {
                'normal': (5, 15),
                'overweight': (15, 25),
                'obese': (25, float('inf'))
            },
            'cat': {
                'normal': (10, 20),
                'overweight': (20, 30),
                'obese': (30, float('inf'))
            },
            'horse': {
                'normal': (8, 18),
                'overweight': (18, 28),
                'obese': (28, float('inf'))
            },
            'cattle': {
                'normal': (10, 25),
                'overweight': (25, 35),
                'obese': (35, float('inf'))
            }
        }
        
        fat_percentage = fat_stats['total_fat_percentage']
        species_thresholds = thresholds.get(species.lower(), thresholds['human'])
        
        # Determine weight category
        if fat_percentage <= species_thresholds['normal'][1]:
            category = "Normal Weight"
            color = "green"
            recommendation = "Maintain current diet and exercise routine."
        elif fat_percentage <= species_thresholds['overweight'][1]:
            category = "Overweight"
            color = "orange"
            recommendation = "Consider dietary adjustments and increased exercise."
        else:
            category = "Obese"
            color = "red"
            recommendation = "Medical consultation recommended for weight management plan."
        
        # Risk assessment based on visceral fat ratio
        vs_ratio = fat_stats['visceral_subcutaneous_ratio']
        if vs_ratio < 0.5:
            risk_level = "Low Risk"
            risk_color = "green"
        elif vs_ratio < 1.0:
            risk_level = "Moderate Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk - Excess visceral fat"
            risk_color = "red"
        
        return {
            'category': category,
            'color': color,
            'recommendation': recommendation,
            'fat_percentage': fat_percentage,
            'visceral_subcutaneous_ratio': vs_ratio,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'species': species
        }


# Convenience functions for integration
def segment_fat(hu_array: np.ndarray, species: str = "human") -> Dict[str, Any]:
    """
    Convenience function for fat segmentation
    
    Args:
        hu_array: 2D array of Hounsfield Unit values
        species: Species type for specific adjustments
        
    Returns:
        Segmentation results dictionary
    """
    engine = FatSegmentationEngine()
    return engine.segment_fat_tissue(hu_array, species)


def create_fat_overlay(base_image: np.ndarray, segmentation_results: Dict[str, Any]) -> np.ndarray:
    """
    Create overlay visualization from segmentation results
    
    Args:
        base_image: Original grayscale image
        segmentation_results: Results from segment_fat function
        
    Returns:
        RGB overlay image
    """
    engine = FatSegmentationEngine()
    engine.fat_mask = segmentation_results.get('fat_mask')
    engine.subcutaneous_mask = segmentation_results.get('subcutaneous_mask')
    engine.visceral_mask = segmentation_results.get('visceral_mask')
    engine.body_outline = segmentation_results.get('body_outline')
    
    return engine.create_overlay_image(base_image)


# Example usage
if __name__ == "__main__":
    print("🔬 Fat Segmentation Engine for Medical Image Analyzer")
    print("Features:")
    print("  - Multi-species support (human, dog, cat, horse, cattle)")
    print("  - HU-based fat tissue detection")
    print("  - Subcutaneous vs visceral fat separation")
    print("  - Species-specific obesity assessment")
    print("  - Clinical risk evaluation")
    print("  - Visual overlay generation")
    print("  - Comprehensive fat statistics")