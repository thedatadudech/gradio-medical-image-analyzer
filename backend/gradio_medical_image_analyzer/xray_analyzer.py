#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X-Ray Image Analyzer for bone and tissue segmentation
Extended for multi-tissue analysis in medical imaging
"""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.exposure import equalize_adapthist
from typing import Dict, Any, Optional, List, Tuple
import cv2


class XRayAnalyzer:
    """Analyze X-Ray images for comprehensive tissue segmentation"""
    
    def __init__(self):
        self.modality_types = ['CR', 'DX', 'RX', 'DR']  # Computed/Digital/Direct Radiography, X-Ray
        self.tissue_types = ['bone', 'soft_tissue', 'air', 'metal', 'fat', 'fluid']
    
    def analyze_xray_image(self, pixel_array: np.ndarray, metadata: dict = None) -> dict:
        """
        Analyze X-Ray image and segment different structures
        
        Args:
            pixel_array: 2D numpy array of pixel values
            metadata: DICOM metadata (optional)
            
        Returns:
            Dictionary with segmentation results
        """
        # Normalize image to 0-1 range
        normalized = self._normalize_image(pixel_array)
        
        # Calculate image statistics
        stats = self._calculate_statistics(normalized)
        
        # Segment different structures with enhanced tissue detection
        segments = {
            'bone': self._segment_bone(normalized, stats),
            'soft_tissue': self._segment_soft_tissue(normalized, stats),
            'air': self._segment_air(normalized, stats),
            'metal': self._detect_metal(normalized, stats),
            'fat': self._segment_fat_xray(normalized, stats),
            'fluid': self._detect_fluid(normalized, stats)
        }
        
        # Calculate percentages
        total_pixels = pixel_array.size
        percentages = {
            name: (np.sum(mask) / total_pixels * 100) 
            for name, mask in segments.items()
        }
        
        # Perform clinical analysis
        clinical_analysis = self._perform_clinical_analysis(segments, stats, metadata)
        
        return {
            'segments': segments,
            'percentages': percentages,
            'statistics': stats,
            'clinical_analysis': clinical_analysis,
            'overlay': self._create_overlay(normalized, segments),
            'tissue_map': self._create_tissue_map(segments)
        }
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min == 0:
            return np.zeros_like(image, dtype=np.float32)
        return (image - img_min) / (img_max - img_min)
    
    def _calculate_statistics(self, image: np.ndarray) -> dict:
        """Calculate comprehensive image statistics for adaptive processing"""
        return {
            'mean': np.mean(image),
            'std': np.std(image),
            'median': np.median(image),
            'skewness': float(np.mean(((image - np.mean(image)) / np.std(image)) ** 3)),
            'kurtosis': float(np.mean(((image - np.mean(image)) / np.std(image)) ** 4) - 3),
            'percentiles': {
                f'p{p}': np.percentile(image, p) 
                for p in [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]
            },
            'histogram': np.histogram(image, bins=256)[0]
        }
    
    def _segment_bone(self, image: np.ndarray, stats: dict) -> np.ndarray:
        """
        Enhanced bone segmentation using multiple techniques
        """
        percentiles = stats['percentiles']
        
        # Method 1: Percentile-based thresholding
        bone_threshold = percentiles['p80']
        bone_mask = image > bone_threshold
        
        # Method 2: Otsu's method on high-intensity regions
        high_intensity = image > percentiles['p60']
        if np.any(high_intensity):
            otsu_thresh = filters.threshold_otsu(image[high_intensity])
            bone_mask_otsu = image > otsu_thresh
            bone_mask = bone_mask | bone_mask_otsu
        
        # Method 3: Gradient-based edge detection for cortical bone
        gradient_magnitude = filters.sobel(image)
        high_gradient = gradient_magnitude > np.percentile(gradient_magnitude, 90)
        bone_edges = high_gradient & (image > percentiles['p70'])
        
        # Combine methods
        bone_mask = bone_mask | bone_edges
        
        # Clean up using morphological operations
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=100)
        bone_mask = morphology.binary_closing(bone_mask, morphology.disk(3))
        bone_mask = morphology.binary_dilation(bone_mask, morphology.disk(1))
        
        return bone_mask.astype(np.uint8)
    
    def _segment_soft_tissue(self, image: np.ndarray, stats: dict) -> np.ndarray:
        """
        Enhanced soft tissue segmentation with better discrimination
        """
        percentiles = stats['percentiles']
        
        # Soft tissue is between air and bone
        soft_lower = percentiles['p20']
        soft_upper = percentiles['p75']
        
        # Initial mask
        soft_mask = (image > soft_lower) & (image < soft_upper)
        
        # Use adaptive thresholding for better edge detection
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Multiple adaptive threshold scales
        adaptive_masks = []
        for block_size in [31, 51, 71]:
            adaptive_thresh = cv2.adaptiveThreshold(
                img_uint8, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=block_size,
                C=2
            )
            adaptive_masks.append(adaptive_thresh > 0)
        
        # Combine adaptive masks
        combined_adaptive = np.logical_and.reduce(adaptive_masks)
        soft_mask = soft_mask & combined_adaptive
        
        # Remove bone and air regions
        bone_mask = self._segment_bone(image, stats)
        air_mask = self._segment_air(image, stats)
        soft_mask = soft_mask & ~bone_mask & ~air_mask
        
        # Clean up
        soft_mask = morphology.remove_small_objects(soft_mask, min_size=300)
        soft_mask = morphology.binary_closing(soft_mask, morphology.disk(2))
        
        return soft_mask.astype(np.uint8)
    
    def _segment_air(self, image: np.ndarray, stats: dict) -> np.ndarray:
        """
        Enhanced air/lung segmentation with better boundary detection
        """
        percentiles = stats['percentiles']
        
        # Air/lung regions are typically very dark
        air_threshold = percentiles['p15']
        air_mask = image < air_threshold
        
        # For chest X-rays, lungs are large dark regions
        # Remove small dark spots (could be noise)
        air_mask = morphology.remove_small_objects(air_mask, min_size=1000)
        
        # Fill holes to get complete lung fields
        air_mask = ndimage.binary_fill_holes(air_mask)
        
        # Refine boundaries using watershed
        distance = ndimage.distance_transform_edt(air_mask)
        local_maxima = morphology.local_maxima(distance)
        markers = measure.label(local_maxima)
        
        if np.any(markers):
            air_mask = morphology.watershed(-distance, markers, mask=air_mask)
            air_mask = air_mask > 0
        
        return air_mask.astype(np.uint8)
    
    def _detect_metal(self, image: np.ndarray, stats: dict) -> np.ndarray:
        """
        Enhanced metal detection including surgical implants
        """
        percentiles = stats['percentiles']
        
        # Metal often saturates the detector
        metal_threshold = percentiles['p99']
        metal_mask = image > metal_threshold
        
        # Check for high local contrast (characteristic of metal)
        local_std = ndimage.generic_filter(image, np.std, size=5)
        high_contrast = local_std > stats['std'] * 2
        
        # Saturation detection - completely white areas
        saturated = image >= 0.99
        
        # Combine criteria
        metal_mask = (metal_mask & high_contrast) | saturated
        
        # Metal objects often have sharp edges
        edges = filters.sobel(image)
        sharp_edges = edges > np.percentile(edges, 95)
        metal_mask = metal_mask | (sharp_edges & (image > percentiles['p95']))
        
        return metal_mask.astype(np.uint8)
    
    def _segment_fat_xray(self, image: np.ndarray, stats: dict) -> np.ndarray:
        """
        Detect fat tissue in X-ray (appears darker than muscle but lighter than air)
        """
        percentiles = stats['percentiles']
        
        # Fat appears darker than soft tissue but lighter than air
        fat_lower = percentiles['p15']
        fat_upper = percentiles['p40']
        
        fat_mask = (image > fat_lower) & (image < fat_upper)
        
        # Fat has relatively uniform texture
        texture = ndimage.generic_filter(image, np.std, size=7)
        low_texture = texture < stats['std'] * 0.5
        
        fat_mask = fat_mask & low_texture
        
        # Remove air regions
        air_mask = self._segment_air(image, stats)
        fat_mask = fat_mask & ~air_mask
        
        # Clean up
        fat_mask = morphology.remove_small_objects(fat_mask, min_size=200)
        fat_mask = morphology.binary_closing(fat_mask, morphology.disk(2))
        
        return fat_mask.astype(np.uint8)
    
    def _detect_fluid(self, image: np.ndarray, stats: dict) -> np.ndarray:
        """
        Detect fluid accumulation (pleural effusion, ascites, etc.)
        """
        percentiles = stats['percentiles']
        
        # Fluid has intermediate density between air and soft tissue
        fluid_lower = percentiles['p25']
        fluid_upper = percentiles['p60']
        
        fluid_mask = (image > fluid_lower) & (image < fluid_upper)
        
        # Fluid tends to accumulate in dependent regions and has smooth boundaries
        # Use gradient to find smooth regions
        gradient = filters.sobel(image)
        smooth_regions = gradient < np.percentile(gradient, 30)
        
        fluid_mask = fluid_mask & smooth_regions
        
        # Fluid collections are usually larger areas
        fluid_mask = morphology.remove_small_objects(fluid_mask, min_size=500)
        
        # Apply closing to fill gaps
        fluid_mask = morphology.binary_closing(fluid_mask, morphology.disk(5))
        
        return fluid_mask.astype(np.uint8)
    
    def _create_tissue_map(self, segments: dict) -> np.ndarray:
        """
        Create a labeled tissue map where each pixel has a tissue type ID
        """
        tissue_map = np.zeros(list(segments.values())[0].shape, dtype=np.uint8)
        
        # Assign tissue IDs (higher priority overwrites lower)
        tissue_priorities = [
            ('air', 1),
            ('fat', 2),
            ('fluid', 3),
            ('soft_tissue', 4),
            ('bone', 5),
            ('metal', 6)
        ]
        
        for tissue_name, tissue_id in tissue_priorities:
            if tissue_name in segments:
                tissue_map[segments[tissue_name] > 0] = tissue_id
        
        return tissue_map
    
    def _create_overlay(self, image: np.ndarray, segments: dict) -> np.ndarray:
        """Create enhanced color overlay visualization"""
        # Convert to RGB
        rgb_image = np.stack([image, image, image], axis=2)
        
        # Enhanced color mappings
        colors = {
            'bone': [1.0, 1.0, 0.8],       # Light yellow
            'soft_tissue': [1.0, 0.7, 0.7],  # Light red
            'air': [0.7, 0.7, 1.0],         # Light blue
            'metal': [1.0, 0.5, 0.0],       # Orange
            'fat': [0.9, 0.9, 0.5],         # Pale yellow
            'fluid': [0.5, 0.8, 1.0]        # Cyan
        }
        
        # Apply colors with transparency
        alpha = 0.3
        for name, mask in segments.items():
            if name in colors and np.any(mask):
                for i in range(3):
                    rgb_image[:, :, i] = np.where(
                        mask,
                        rgb_image[:, :, i] * (1 - alpha) + colors[name][i] * alpha,
                        rgb_image[:, :, i]
                    )
        
        return rgb_image
    
    def _perform_clinical_analysis(self, segments: dict, stats: dict, 
                                  metadata: Optional[dict]) -> dict:
        """
        Perform clinical analysis based on segmentation results
        """
        analysis = {
            'tissue_distribution': self._analyze_tissue_distribution(segments),
            'abnormality_detection': self._detect_abnormalities(segments, stats),
            'quality_assessment': self._assess_image_quality(stats)
        }
        
        # Add body-part specific analysis if metadata available
        if metadata and 'BodyPartExamined' in metadata:
            body_part = metadata['BodyPartExamined'].lower()
            analysis['body_part_analysis'] = self._analyze_body_part(
                segments, stats, body_part
            )
        
        return analysis
    
    def _analyze_tissue_distribution(self, segments: dict) -> dict:
        """Analyze the distribution of different tissues"""
        total_pixels = list(segments.values())[0].size
        
        distribution = {}
        for tissue, mask in segments.items():
            pixels = np.sum(mask)
            percentage = (pixels / total_pixels) * 100
            distribution[tissue] = {
                'pixels': int(pixels),
                'percentage': round(percentage, 2),
                'present': pixels > 100  # Minimum threshold
            }
        
        # Calculate ratios
        if distribution['soft_tissue']['pixels'] > 0:
            distribution['bone_to_soft_ratio'] = round(
                distribution['bone']['pixels'] / distribution['soft_tissue']['pixels'], 
                3
            )
        
        return distribution
    
    def _detect_abnormalities(self, segments: dict, stats: dict) -> dict:
        """Detect potential abnormalities in the image"""
        abnormalities = {
            'detected': False,
            'findings': []
        }
        
        # Check for unusual tissue distributions
        tissue_dist = self._analyze_tissue_distribution(segments)
        
        # High metal content might indicate implants
        if tissue_dist['metal']['percentage'] > 0.5:
            abnormalities['detected'] = True
            abnormalities['findings'].append({
                'type': 'metal_implant',
                'confidence': 'high',
                'description': 'Metal implant or foreign body detected'
            })
        
        # Fluid accumulation
        if tissue_dist['fluid']['percentage'] > 5:
            abnormalities['detected'] = True
            abnormalities['findings'].append({
                'type': 'fluid_accumulation',
                'confidence': 'medium',
                'description': 'Possible fluid accumulation detected'
            })
        
        # Asymmetry detection for bilateral structures
        if 'air' in segments:
            asymmetry = self._check_bilateral_symmetry(segments['air'])
            if asymmetry > 0.3:  # 30% asymmetry threshold
                abnormalities['detected'] = True
                abnormalities['findings'].append({
                    'type': 'asymmetry',
                    'confidence': 'medium',
                    'description': f'Bilateral asymmetry detected ({asymmetry:.1%})'
                })
        
        return abnormalities
    
    def _check_bilateral_symmetry(self, mask: np.ndarray) -> float:
        """Check symmetry of bilateral structures"""
        height, width = mask.shape
        left_half = mask[:, :width//2]
        right_half = mask[:, width//2:]
        
        # Flip right half for comparison
        right_half_flipped = np.fliplr(right_half)
        
        # Calculate difference
        if right_half_flipped.shape[1] != left_half.shape[1]:
            # Handle odd width
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
        
        difference = np.sum(np.abs(left_half.astype(float) - right_half_flipped.astype(float)))
        total = np.sum(left_half) + np.sum(right_half_flipped)
        
        if total == 0:
            return 0.0
        
        return difference / total
    
    def _assess_image_quality(self, stats: dict) -> dict:
        """Assess the quality of the X-ray image"""
        quality = {
            'overall': 'good',
            'issues': []
        }
        
        # Check contrast
        if stats['std'] < 0.1:
            quality['overall'] = 'poor'
            quality['issues'].append('Low contrast')
        
        # Check if image is too dark or too bright
        if stats['mean'] < 0.2:
            quality['overall'] = 'fair' if quality['overall'] == 'good' else 'poor'
            quality['issues'].append('Underexposed')
        elif stats['mean'] > 0.8:
            quality['overall'] = 'fair' if quality['overall'] == 'good' else 'poor'
            quality['issues'].append('Overexposed')
        
        # Check histogram distribution
        hist = stats['histogram']
        if np.max(hist) > 0.5 * np.sum(hist):
            quality['overall'] = 'fair' if quality['overall'] == 'good' else 'poor'
            quality['issues'].append('Poor histogram distribution')
        
        return quality
    
    def _analyze_body_part(self, segments: dict, stats: dict, body_part: str) -> dict:
        """Perform body-part specific analysis"""
        if 'chest' in body_part or 'thorax' in body_part:
            return self._analyze_chest_xray(segments, stats)
        elif 'abdom' in body_part:
            return self._analyze_abdominal_xray(segments, stats)
        elif 'extrem' in body_part or 'limb' in body_part:
            return self._analyze_extremity_xray(segments, stats)
        else:
            return {'body_part': body_part, 'analysis': 'Generic analysis performed'}
    
    def _analyze_chest_xray(self, segments: dict, stats: dict) -> dict:
        """Specific analysis for chest X-rays"""
        analysis = {
            'lung_fields': 'not_assessed',
            'cardiac_size': 'not_assessed',
            'mediastinum': 'not_assessed'
        }
        
        # Analyze lung fields
        if 'air' in segments:
            air_mask = segments['air']
            labeled = measure.label(air_mask)
            regions = measure.regionprops(labeled)
            
            large_regions = [r for r in regions if r.area > 1000]
            if len(large_regions) >= 2:
                analysis['lung_fields'] = 'bilateral_present'
                # Check symmetry
                symmetry = self._check_bilateral_symmetry(air_mask)
                if symmetry < 0.2:
                    analysis['lung_symmetry'] = 'symmetric'
                else:
                    analysis['lung_symmetry'] = 'asymmetric'
            elif len(large_regions) == 1:
                analysis['lung_fields'] = 'unilateral_present'
            else:
                analysis['lung_fields'] = 'not_visualized'
        
        # Analyze cardiac silhouette
        if 'soft_tissue' in segments:
            soft_mask = segments['soft_tissue']
            height, width = soft_mask.shape
            central_region = soft_mask[height//3:2*height//3, width//3:2*width//3]
            
            if np.any(central_region):
                analysis['cardiac_size'] = 'present'
                # Simple cardiothoracic ratio estimation
                cardiac_width = np.sum(np.any(central_region, axis=0))
                thoracic_width = width
                ctr = cardiac_width / thoracic_width
                analysis['cardiothoracic_ratio'] = round(ctr, 2)
                
                if ctr > 0.5:
                    analysis['cardiac_assessment'] = 'enlarged'
                else:
                    analysis['cardiac_assessment'] = 'normal'
        
        return analysis
    
    def _analyze_abdominal_xray(self, segments: dict, stats: dict) -> dict:
        """Specific analysis for abdominal X-rays"""
        analysis = {
            'gas_pattern': 'not_assessed',
            'soft_tissue_masses': 'not_assessed',
            'calcifications': 'not_assessed'
        }
        
        # Analyze gas patterns
        if 'air' in segments:
            air_mask = segments['air']
            labeled = measure.label(air_mask)
            regions = measure.regionprops(labeled)
            
            small_gas = [r for r in regions if 50 < r.area < 500]
            large_gas = [r for r in regions if r.area >= 500]
            
            if len(small_gas) > 10:
                analysis['gas_pattern'] = 'increased_small_bowel_gas'
            elif len(large_gas) > 3:
                analysis['gas_pattern'] = 'distended_bowel_loops'
            else:
                analysis['gas_pattern'] = 'normal'
        
        # Check for calcifications (bright spots)
        if 'bone' in segments:
            bone_mask = segments['bone']
            labeled = measure.label(bone_mask)
            regions = measure.regionprops(labeled)
            
            small_bright = [r for r in regions if r.area < 50]
            if len(small_bright) > 5:
                analysis['calcifications'] = 'present'
            else:
                analysis['calcifications'] = 'none_detected'
        
        return analysis
    
    def _analyze_extremity_xray(self, segments: dict, stats: dict) -> dict:
        """Specific analysis for extremity X-rays"""
        analysis = {
            'bone_integrity': 'not_assessed',
            'joint_spaces': 'not_assessed',
            'soft_tissue_swelling': 'not_assessed'
        }
        
        # Analyze bone continuity
        if 'bone' in segments:
            bone_mask = segments['bone']
            
            # Check for discontinuities (potential fractures)
            skeleton = morphology.skeletonize(bone_mask)
            endpoints = self._find_endpoints(skeleton)
            
            if len(endpoints) > 4:  # More than expected endpoints
                analysis['bone_integrity'] = 'possible_discontinuity'
            else:
                analysis['bone_integrity'] = 'continuous'
            
            # Analyze bone density
            analysis['bone_density'] = self._assess_bone_density_pattern(bone_mask)
        
        # Check for soft tissue swelling
        if 'soft_tissue' in segments:
            soft_mask = segments['soft_tissue']
            soft_area = np.sum(soft_mask)
            total_area = soft_mask.size
            
            soft_percentage = (soft_area / total_area) * 100
            if soft_percentage > 40:
                analysis['soft_tissue_swelling'] = 'increased'
            else:
                analysis['soft_tissue_swelling'] = 'normal'
        
        return analysis
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in a skeletonized image"""
        endpoints = []
        
        # Define 8-connectivity kernel
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Find points with only one neighbor
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j]:
                    neighbors = np.sum(kernel * skeleton[i-1:i+2, j-1:j+2])
                    if neighbors == 1:
                        endpoints.append((i, j))
        
        return endpoints
    
    def _assess_bone_density_pattern(self, bone_mask: np.ndarray) -> str:
        """Assess bone density patterns"""
        # Simple assessment based on coverage
        bone_pixels = np.sum(bone_mask)
        total_pixels = bone_mask.size
        coverage = bone_pixels / total_pixels
        
        if coverage > 0.15:
            return 'normal'
        elif coverage > 0.10:
            return 'mildly_decreased'
        else:
            return 'significantly_decreased'
    
    def classify_pixel(self, pixel_value: float, x: int, y: int, 
                      image_array: np.ndarray) -> dict:
        """
        Enhanced pixel classification with spatial context
        
        Args:
            pixel_value: Normalized pixel value (0-1)
            x, y: Pixel coordinates
            image_array: Full image array for context
            
        Returns:
            Classification result with confidence
        """
        # Get local statistics
        window_size = 5
        half_window = window_size // 2
        
        # Ensure we don't go out of bounds
        x_start = max(0, x - half_window)
        x_end = min(image_array.shape[1], x + half_window + 1)
        y_start = max(0, y - half_window)
        y_end = min(image_array.shape[0], y + half_window + 1)
        
        local_region = image_array[y_start:y_end, x_start:x_end]
        local_mean = np.mean(local_region)
        local_std = np.std(local_region)
        
        # Calculate image statistics if not provided
        image_stats = self._calculate_statistics(image_array)
        percentiles = image_stats['percentiles']
        
        # Enhanced classification with confidence
        if pixel_value > percentiles['p95']:
            tissue_type = 'Metal/Implant'
            icon = '⚙️'
            color = '#FFA500'
            confidence = 'high' if pixel_value > percentiles['p99'] else 'medium'
        elif pixel_value > percentiles['p80']:
            tissue_type = 'Bone'
            icon = '🦴'
            color = '#FFFACD'
            # Check local texture for bone confidence
            if local_std > image_stats['std'] * 0.8:
                confidence = 'high'
            else:
                confidence = 'medium'
        elif pixel_value > percentiles['p60']:
            tissue_type = 'Dense Soft Tissue'
            icon = '💪'
            color = '#FFB6C1'
            confidence = 'medium'
        elif pixel_value > percentiles['p40']:
            tissue_type = 'Soft Tissue'
            icon = '🔴'
            color = '#FFC0CB'
            confidence = 'high' if local_std < image_stats['std'] * 0.5 else 'medium'
        elif pixel_value > percentiles['p20']:
            # Could be fat or fluid
            if local_std < image_stats['std'] * 0.3:
                tissue_type = 'Fat'
                icon = '🟡'
                color = '#FFFFE0'
            else:
                tissue_type = 'Fluid'
                icon = '💧'
                color = '#87CEEB'
            confidence = 'medium'
        else:
            tissue_type = 'Air/Lung'
            icon = '🌫️'
            color = '#ADD8E6'
            confidence = 'high' if pixel_value < percentiles['p10'] else 'medium'
        
        return {
            'type': tissue_type,
            'icon': icon,
            'color': color,
            'confidence': confidence,
            'pixel_value': round(pixel_value, 3),
            'local_context': {
                'mean': round(local_mean, 3),
                'std': round(local_std, 3)
            }
        }


# Convenience functions for integration
def analyze_xray(pixel_array: np.ndarray, body_part: Optional[str] = None, 
                metadata: Optional[dict] = None) -> dict:
    """
    Convenience function for X-ray analysis
    
    Args:
        pixel_array: X-ray image array
        body_part: Optional body part specification
        metadata: Optional DICOM metadata
        
    Returns:
        Comprehensive analysis results
    """
    analyzer = XRayAnalyzer()
    results = analyzer.analyze_xray_image(pixel_array, metadata)
    
    # Add body part specific analysis if specified
    if body_part and 'clinical_analysis' in results:
        results['clinical_analysis']['body_part_analysis'] = analyzer._analyze_body_part(
            results['segments'], results['statistics'], body_part
        )
    
    return results


def classify_xray_tissue(pixel_value: float, x: int, y: int, 
                        image_array: np.ndarray) -> dict:
    """
    Convenience function for tissue classification at a specific pixel
    
    Args:
        pixel_value: Normalized pixel value
        x, y: Pixel coordinates
        image_array: Full image for context
        
    Returns:
        Tissue classification result
    """
    analyzer = XRayAnalyzer()
    return analyzer.classify_pixel(pixel_value, x, y, image_array)


# Example usage
if __name__ == "__main__":
    print("🔬 Advanced X-Ray Analyzer for Medical Image Analysis")
    print("Features:")
    print("  - Multi-tissue segmentation (bone, soft tissue, air, metal, fat, fluid)")
    print("  - Clinical abnormality detection")
    print("  - Body-part specific analysis (chest, abdomen, extremity)")
    print("  - Image quality assessment")
    print("  - Spatial context-aware tissue classification")
    print("  - Symmetry and structural analysis")
    print("  - Comprehensive statistical analysis")