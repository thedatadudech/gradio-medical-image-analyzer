#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Custom Component: Medical Image Analyzer
AI-Agent optimized component for medical image analysis
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import gradio as gr
from gradio.components.base import Component
from gradio.events import Events
import numpy as np
import json

# Import our existing analyzers
try:
    from .fat_segmentation import FatSegmentationEngine, segment_fat, create_fat_overlay
    FAT_SEGMENTATION_AVAILABLE = True
except ImportError:
    FAT_SEGMENTATION_AVAILABLE = False
    FatSegmentationEngine = None
    segment_fat = None
    create_fat_overlay = None

try:
    from .xray_analyzer import XRayAnalyzer, analyze_xray, classify_xray_tissue
    XRAY_ANALYZER_AVAILABLE = True
except ImportError:
    XRAY_ANALYZER_AVAILABLE = False
    XRayAnalyzer = None
    analyze_xray = None
    classify_xray_tissue = None

from gradio.events import Dependency

class MedicalImageAnalyzer(Component):
    """
    A Gradio component for AI-agent compatible medical image analysis.
    
    Provides structured output for:
    - HU value analysis (CT only)
    - Tissue classification
    - Fat segmentation (subcutaneous, visceral)
    - Confidence scores and reasoning
    """
    
    EVENTS = [
        Events.change,
        Events.select,
        Events.upload,
        Events.clear,
    ]
    
    # HU ranges for tissue classification (CT only)
    HU_RANGES = {
        'air': {'min': -1000, 'max': -500, 'icon': '🌫️'},
        'fat': {'min': -100, 'max': -50, 'icon': '🟡'},
        'water': {'min': -10, 'max': 10, 'icon': '💧'},
        'soft_tissue': {'min': 30, 'max': 80, 'icon': '🔴'},
        'bone': {'min': 200, 'max': 3000, 'icon': '🦴'}
    }
    
    def __init__(
        self,
        value: Optional[Dict[str, Any]] = None,
        *,
        label: Optional[str] = None,
        info: Optional[str] = None,
        every: Optional[float] = None,
        show_label: Optional[bool] = None,
        container: Optional[bool] = None,
        scale: Optional[int] = None,
        min_width: Optional[int] = None,
        visible: Optional[bool] = None,
        elem_id: Optional[str] = None,
        elem_classes: Optional[List[str] | str] = None,
        render: Optional[bool] = None,
        key: Optional[int | str] = None,
        # Custom parameters
        analysis_mode: str = "structured",  # "structured" for agents, "visual" for humans
        include_confidence: bool = True,
        include_reasoning: bool = True,
        segmentation_types: List[str] = None,
        **kwargs,
    ):
        """
        Initialize the Medical Image Analyzer component.
        
        Args:
            analysis_mode: "structured" for AI agents, "visual" for human interpretation
            include_confidence: Include confidence scores in results
            include_reasoning: Include reasoning/explanation for findings
            segmentation_types: List of segmentation types to perform
        """
        self.analysis_mode = analysis_mode
        self.include_confidence = include_confidence
        self.include_reasoning = include_reasoning
        self.segmentation_types = segmentation_types or ["fat_total", "fat_subcutaneous", "fat_visceral"]
        
        if FAT_SEGMENTATION_AVAILABLE:
            self.fat_engine = FatSegmentationEngine()
        else:
            self.fat_engine = None
            
        super().__init__(
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            value=value,
            **kwargs,
        )
    
    def preprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input from frontend.
        Expected format:
        {
            "image": numpy array or base64,
            "modality": "CT" or "XR",
            "pixel_spacing": [x, y] (optional),
            "roi": {"x": int, "y": int, "radius": int} (optional),
            "task": "analyze_point" | "segment_fat" | "full_analysis"
        }
        """
        if payload is None:
            return None
            
        # Validate required fields
        if "image" not in payload:
            return {"error": "No image provided"}
            
        return payload
    
    def postprocess(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess output for frontend.
        Returns structured data for AI agents or formatted HTML for visual mode.
        """
        if value is None:
            return None
            
        if "error" in value:
            return value
            
        # For visual mode, convert to HTML
        if self.analysis_mode == "visual":
            value["html_report"] = self._create_html_report(value)
            
        return value
    
    def analyze_image(
        self, 
        image: np.ndarray,
        modality: str = "CT",
        pixel_spacing: Optional[List[float]] = None,
        roi: Optional[Dict[str, int]] = None,
        task: str = "full_analysis",
        clinical_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main analysis function for medical images.
        
        Args:
            image: 2D numpy array of pixel values
            modality: "CT" or "XR" (X-Ray)
            pixel_spacing: [x, y] spacing in mm
            roi: Region of interest {"x": int, "y": int, "radius": int}
            task: Analysis task
            clinical_context: Additional context for guided analysis
            
        Returns:
            Structured analysis results
        """
        # Handle None or invalid image
        if image is None:
            return {"error": "No image provided", "modality": modality}
            
        results = {
            "modality": modality,
            "timestamp": self._get_timestamp(),
            "measurements": {},
            "findings": [],
            "segmentation": {},
            "quality_metrics": self._assess_image_quality(image)
        }
        
        if task == "analyze_point" and roi:
            # Analyze specific point/region
            results["point_analysis"] = self._analyze_roi(image, modality, roi)
            
        elif task == "segment_fat" and modality == "CT":
            # Fat segmentation
            results["segmentation"] = self._perform_fat_segmentation(image, pixel_spacing)
            
        elif task == "full_analysis":
            # Complete analysis
            if roi:
                results["point_analysis"] = self._analyze_roi(image, modality, roi)
            if modality == "CT":
                results["segmentation"] = self._perform_fat_segmentation(image, pixel_spacing)
            elif modality in ["CR", "DX", "RX", "DR", "X-Ray"]:
                results["segmentation"] = self._perform_xray_analysis(image)
            results["statistics"] = self._calculate_image_statistics(image)
            
        # Add clinical interpretation if context provided
        if clinical_context:
            results["clinical_correlation"] = self._correlate_with_clinical(
                results, clinical_context
            )
            
        return results
    
    def _analyze_roi(self, image: np.ndarray, modality: str, roi: Dict[str, int]) -> Dict[str, Any]:
        """Analyze a specific region of interest"""
        x, y, radius = roi.get("x", 0), roi.get("y", 0), roi.get("radius", 5)
        
        # Extract ROI
        y_min = max(0, y - radius)
        y_max = min(image.shape[0], y + radius)
        x_min = max(0, x - radius)
        x_max = min(image.shape[1], x + radius)
        
        roi_pixels = image[y_min:y_max, x_min:x_max]
        
        analysis = {
            "location": {"x": x, "y": y},
            "roi_size": {"width": x_max - x_min, "height": y_max - y_min},
            "statistics": {
                "mean": float(np.mean(roi_pixels)),
                "std": float(np.std(roi_pixels)),
                "min": float(np.min(roi_pixels)),
                "max": float(np.max(roi_pixels))
            }
        }
        
        if modality == "CT":
            # HU-based analysis
            center_value = float(image[y, x])
            analysis["hu_value"] = center_value
            analysis["tissue_type"] = self._classify_tissue_by_hu(center_value)
            
            if self.include_confidence:
                analysis["confidence"] = self._calculate_confidence(roi_pixels, center_value)
                
            if self.include_reasoning:
                analysis["reasoning"] = self._generate_reasoning(
                    center_value, analysis["tissue_type"], roi_pixels
                )
        else:
            # Intensity-based analysis for X-Ray
            analysis["intensity"] = float(image[y, x])
            analysis["tissue_type"] = self._classify_xray_intensity(
                image[y, x], x, y, image
            )
            
        return analysis
    
    def _classify_tissue_by_hu(self, hu_value: float) -> Dict[str, str]:
        """Classify tissue type based on HU value"""
        for tissue_type, ranges in self.HU_RANGES.items():
            if ranges['min'] <= hu_value <= ranges['max']:
                return {
                    'type': tissue_type,
                    'icon': ranges['icon'],
                    'hu_range': f"{ranges['min']} to {ranges['max']}"
                }
        
        # Edge cases
        if hu_value < -1000:
            return {'type': 'air', 'icon': '🌫️', 'hu_range': '< -1000'}
        else:
            return {'type': 'metal/artifact', 'icon': '⚙️', 'hu_range': '> 3000'}
    
    def _classify_xray_intensity(self, intensity: float, x: int, y: int, image: np.ndarray) -> Dict[str, str]:
        """Classify tissue in X-Ray based on intensity"""
        if XRAY_ANALYZER_AVAILABLE:
            # Use the advanced XRay analyzer
            analyzer = XRayAnalyzer()
            # Normalize the intensity  
            stats = self._calculate_image_statistics(image)
            normalized = (intensity - stats['min']) / (stats['max'] - stats['min'])
            result = analyzer.classify_pixel(normalized, x, y, image)
            return {
                'type': result['type'],
                'icon': result['icon'],
                'confidence': result['confidence'],
                'color': result['color']
            }
        else:
            # Fallback to simple classification
            normalized = (intensity - stats['min']) / (stats['max'] - stats['min'])
            
            if normalized > 0.9:
                return {'type': 'bone/metal', 'icon': '🦴', 'intensity_range': 'very high'}
            elif normalized > 0.6:
                return {'type': 'bone', 'icon': '🦴', 'intensity_range': 'high'}
            elif normalized > 0.3:
                return {'type': 'soft_tissue', 'icon': '🔴', 'intensity_range': 'medium'}
            else:
                return {'type': 'air/lung', 'icon': '🌫️', 'intensity_range': 'low'}
    
    def _perform_fat_segmentation(
        self, 
        image: np.ndarray, 
        pixel_spacing: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Perform fat segmentation using our existing engine"""
        if not self.fat_engine:
            return {"error": "Fat segmentation not available"}
        
        # Use existing fat segmentation
        segmentation_result = self.fat_engine.segment_fat_tissue(image)
        
        results = {
            "statistics": segmentation_result.get("statistics", {}),
            "segments": {}
        }
        
        # Calculate areas if pixel spacing provided
        if pixel_spacing:
            pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
            for segment_type in ["total", "subcutaneous", "visceral"]:
                pixel_key = f"{segment_type}_fat_pixels"
                if pixel_key in results["statistics"]:
                    area_mm2 = results["statistics"][pixel_key] * pixel_area_mm2
                    results["segments"][segment_type] = {
                        "pixels": results["statistics"][pixel_key],
                        "area_mm2": area_mm2,
                        "area_cm2": area_mm2 / 100
                    }
        
        # Add interpretation
        if "total_fat_percentage" in results["statistics"]:
            results["interpretation"] = self._interpret_fat_results(
                results["statistics"]
            )
            
        return results
    
    def _interpret_fat_results(self, stats: Dict[str, float]) -> Dict[str, Any]:
        """Interpret fat segmentation results"""
        interpretation = {
            "obesity_risk": "normal",
            "visceral_risk": "normal",
            "recommendations": []
        }
        
        total_fat = stats.get("total_fat_percentage", 0)
        visceral_ratio = stats.get("visceral_subcutaneous_ratio", 0)
        
        # Obesity assessment
        if total_fat > 40:
            interpretation["obesity_risk"] = "severe"
            interpretation["recommendations"].append("Immediate weight management required")
        elif total_fat > 30:
            interpretation["obesity_risk"] = "moderate"
            interpretation["recommendations"].append("Weight reduction recommended")
        elif total_fat > 25:
            interpretation["obesity_risk"] = "mild"
            interpretation["recommendations"].append("Monitor weight trend")
            
        # Visceral fat assessment
        if visceral_ratio > 0.5:
            interpretation["visceral_risk"] = "high"
            interpretation["recommendations"].append("High visceral fat - metabolic risk")
        elif visceral_ratio > 0.3:
            interpretation["visceral_risk"] = "moderate"
            
        return interpretation
    
    def _perform_xray_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive X-ray analysis using XRayAnalyzer"""
        if not XRAY_ANALYZER_AVAILABLE:
            return {"error": "X-ray analysis not available"}
        
        # Use the XRay analyzer
        analysis_results = analyze_xray(image)
        
        results = {
            "segments": {},
            "tissue_distribution": {},
            "clinical_findings": []
        }
        
        # Process segmentation results
        if "segments" in analysis_results:
            for tissue_type, mask in analysis_results["segments"].items():
                if np.any(mask):
                    pixel_count = np.sum(mask)
                    percentage = analysis_results["percentages"].get(tissue_type, 0)
                    results["segments"][tissue_type] = {
                        "pixels": int(pixel_count),
                        "percentage": round(percentage, 2),
                        "present": True
                    }
        
        # Add tissue distribution
        if "percentages" in analysis_results:
            results["tissue_distribution"] = analysis_results["percentages"]
        
        # Add clinical analysis
        if "clinical_analysis" in analysis_results:
            clinical = analysis_results["clinical_analysis"]
            
            # Quality assessment
            if "quality_assessment" in clinical:
                results["quality"] = clinical["quality_assessment"]
            
            # Abnormality detection
            if "abnormality_detection" in clinical:
                abnorm = clinical["abnormality_detection"]
                if abnorm.get("detected", False):
                    for finding in abnorm.get("findings", []):
                        results["clinical_findings"].append({
                            "type": finding.get("type", "unknown"),
                            "description": finding.get("description", ""),
                            "confidence": finding.get("confidence", "low")
                        })
            
            # Tissue distribution analysis
            if "tissue_distribution" in clinical:
                dist = clinical["tissue_distribution"]
                if "bone_to_soft_ratio" in dist:
                    results["bone_soft_ratio"] = dist["bone_to_soft_ratio"]
        
        # Add interpretation
        results["interpretation"] = self._interpret_xray_results(results)
        
        return results
    
    def _interpret_xray_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret X-ray analysis results"""
        interpretation = {
            "summary": "Normal X-ray appearance",
            "findings": [],
            "recommendations": []
        }
        
        # Check for abnormal findings
        if results.get("clinical_findings"):
            interpretation["summary"] = "Abnormal findings detected"
            for finding in results["clinical_findings"]:
                interpretation["findings"].append(finding["description"])
        
        # Check tissue distribution
        tissue_dist = results.get("tissue_distribution", {})
        if tissue_dist.get("metal", 0) > 0.5:
            interpretation["findings"].append("Metal artifact/implant present")
        
        if tissue_dist.get("fluid", 0) > 5:
            interpretation["findings"].append("Possible fluid accumulation")
            interpretation["recommendations"].append("Clinical correlation recommended")
        
        # Check quality
        quality = results.get("quality", {})
        if quality.get("overall") in ["poor", "fair"]:
            interpretation["recommendations"].append("Consider repeat imaging for better quality")
        
        return interpretation
    
    def _calculate_confidence(self, roi_pixels: np.ndarray, center_value: float) -> float:
        """Calculate confidence score based on ROI homogeneity"""
        if roi_pixels.size == 0:
            return 0.0
        
        # Handle single pixel or uniform regions
        if roi_pixels.size == 1 or np.all(roi_pixels == roi_pixels.flat[0]):
            return 1.0  # Perfect confidence for uniform regions
            
        # Check how consistent the ROI is
        std = np.std(roi_pixels)
        mean = np.mean(roi_pixels)
        
        # Handle zero std (uniform region)
        if std == 0:
            return 1.0
        
        # How close is center value to mean
        center_deviation = abs(center_value - mean) / std
        
        # Coefficient of variation (normalized)
        cv = std / (abs(mean) + 1e-6)
        
        # Base confidence from homogeneity (sigmoid-like transformation)
        # CV of 0.1 = ~95% confidence, CV of 0.5 = ~70% confidence, CV of 1.0 = ~50% confidence
        base_confidence = 1.0 / (1.0 + cv * 2.0)
        
        # Adjust based on center deviation
        # If center is close to mean, increase confidence
        deviation_factor = 1.0 / (1.0 + center_deviation * 0.5)
        
        # Combine factors
        confidence = base_confidence * 0.7 + deviation_factor * 0.3
        
        # Ensure reasonable minimum confidence for valid detections
        confidence = max(0.5, min(0.99, confidence))
        
        return round(confidence, 2)
    
    def _generate_reasoning(
        self, 
        hu_value: float, 
        tissue_type: Dict[str, str], 
        roi_pixels: np.ndarray
    ) -> str:
        """Generate reasoning for the classification"""
        reasoning_parts = []
        
        # HU value interpretation
        reasoning_parts.append(f"HU value of {hu_value:.1f} falls within {tissue_type['type']} range")
        
        # Homogeneity assessment
        std = np.std(roi_pixels)
        if std < 10:
            reasoning_parts.append("Homogeneous region suggests uniform tissue")
        elif std < 30:
            reasoning_parts.append("Moderate heterogeneity observed")
        else:
            reasoning_parts.append("High heterogeneity - possible mixed tissues or pathology")
            
        return ". ".join(reasoning_parts)
    
    def _calculate_image_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive image statistics"""
        return {
            "min": float(np.min(image)),
            "max": float(np.max(image)),
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "median": float(np.median(image)),
            "p5": float(np.percentile(image, 5)),
            "p95": float(np.percentile(image, 95))
        }
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality metrics"""
        # Simple quality metrics
        dynamic_range = np.max(image) - np.min(image)
        snr = np.mean(image) / (np.std(image) + 1e-6)
        
        quality = {
            "dynamic_range": float(dynamic_range),
            "snr": float(snr),
            "assessment": "good" if snr > 10 and dynamic_range > 100 else "poor"
        }
        
        return quality
    
    def _correlate_with_clinical(
        self, 
        analysis_results: Dict[str, Any], 
        clinical_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate findings with clinical context"""
        correlation = {
            "relevant_findings": [],
            "clinical_significance": "unknown"
        }
        
        # Example correlation logic
        if "symptoms" in clinical_context:
            symptoms = clinical_context["symptoms"]
            
            if "dyspnea" in symptoms and analysis_results.get("modality") == "CT":
                # Check for lung pathology indicators
                if "segmentation" in analysis_results:
                    fat_percent = analysis_results["segmentation"]["statistics"].get(
                        "total_fat_percentage", 0
                    )
                    if fat_percent > 35:
                        correlation["relevant_findings"].append(
                            "High body fat may contribute to dyspnea"
                        )
                        
        return correlation
    
    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML report for visual mode"""
        html_parts = ['<div class="medical-analysis-report">']
        
        # Header
        html_parts.append(f'<h3>Medical Image Analysis Report</h3>')
        html_parts.append(f'<p><strong>Modality:</strong> {results.get("modality", "Unknown")}</p>')
        
        # Point analysis
        if "point_analysis" in results:
            pa = results["point_analysis"]
            html_parts.append('<div class="point-analysis">')
            html_parts.append('<h4>Point Analysis</h4>')
            
            if "hu_value" in pa:
                html_parts.append(f'<p>HU Value: {pa["hu_value"]:.1f}</p>')
            
            tissue = pa.get("tissue_type", {})
            html_parts.append(
                f'<p>Tissue Type: {tissue.get("icon", "")} {tissue.get("type", "Unknown")}</p>'
            )
            
            if "confidence" in pa:
                html_parts.append(f'<p>Confidence: {pa["confidence"]*100:.0f}%</p>')
                
            if "reasoning" in pa:
                html_parts.append(f'<p><em>{pa["reasoning"]}</em></p>')
                
            html_parts.append('</div>')
        
        # Segmentation results
        if "segmentation" in results and "statistics" in results["segmentation"]:
            stats = results["segmentation"]["statistics"]
            html_parts.append('<div class="segmentation-results">')
            html_parts.append('<h4>Fat Segmentation Analysis</h4>')
            html_parts.append(f'<p>Total Fat: {stats.get("total_fat_percentage", 0):.1f}%</p>')
            html_parts.append(f'<p>Subcutaneous: {stats.get("subcutaneous_fat_percentage", 0):.1f}%</p>')
            html_parts.append(f'<p>Visceral: {stats.get("visceral_fat_percentage", 0):.1f}%</p>')
            
            if "interpretation" in results["segmentation"]:
                interp = results["segmentation"]["interpretation"]
                html_parts.append(f'<p><strong>Risk:</strong> {interp.get("obesity_risk", "normal")}</p>')
                
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        # Add CSS
        html_parts.insert(0, '''<style>
        .medical-analysis-report {
            font-family: Arial, sans-serif;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .medical-analysis-report h3, .medical-analysis-report h4 {
            color: #2c3e50;
            margin-top: 10px;
        }
        .point-analysis, .segmentation-results {
            background: white;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>''')
        
        return ''.join(html_parts)
    
    def process_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process uploaded file (DICOM or regular image)
        
        Returns:
            pixel_array: numpy array of pixel values
            display_array: normalized array for display (0-255)
            metadata: file metadata including modality
        """
        if not file_path:
            raise ValueError("No file provided")
        
        file_ext = Path(file_path).suffix.lower()
        
        # Try DICOM first - always try to read as DICOM regardless of extension
        if PYDICOM_AVAILABLE:
            try:
                ds = pydicom.dcmread(file_path, force=True)
                
                # Extract pixel array
                pixel_array = ds.pixel_array.astype(float)
                
                # Get modality
                modality = ds.get('Modality', 'CT')
                
                # Apply DICOM transformations
                if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                
                # Normalize for display
                if modality == 'CT':
                    # CT: typically -1000 to 3000 HU
                    display_array = np.clip((pixel_array + 1000) / 4000 * 255, 0, 255).astype(np.uint8)
                else:
                    # X-ray: normalize to full range
                    pmin, pmax = np.percentile(pixel_array, [1, 99])
                    display_array = np.clip((pixel_array - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)
                
                metadata = {
                    'modality': modality,
                    'shape': pixel_array.shape,
                    'patient_name': str(ds.get('PatientName', 'Anonymous')),
                    'study_date': str(ds.get('StudyDate', '')),
                    'file_type': 'DICOM'
                }
                
                if 'WindowCenter' in ds and 'WindowWidth' in ds:
                    metadata['window_center'] = float(ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0])
                    metadata['window_width'] = float(ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0])
                
                return pixel_array, display_array, metadata
                
            except Exception:
                # If DICOM reading fails, try as regular image
                pass
        
        # Handle regular images
        try:
            img = Image.open(file_path)
            
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            pixel_array = np.array(img).astype(float)
            display_array = pixel_array.astype(np.uint8)
            
            # Guess modality from filename
            filename_lower = Path(file_path).name.lower()
            if 'ct' in filename_lower:
                modality = 'CT'
            else:
                modality = 'CR'  # Default to X-ray
            
            metadata = {
                'modality': modality,
                'shape': pixel_array.shape,
                'file_type': 'Image',
                'format': img.format
            }
            
            return pixel_array, display_array, metadata
            
        except Exception as e:
            raise ValueError(f"Could not load file: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def api_info(self) -> Dict[str, Any]:
        """Return API information for the component"""
        return {
            "info": {
                "type": "object",
                "description": "Medical image analysis results",
                "properties": {
                    "modality": {"type": "string"},
                    "measurements": {"type": "object"},
                    "findings": {"type": "array"},
                    "segmentation": {"type": "object"},
                    "quality_metrics": {"type": "object"}
                }
            },
            "serialized_info": True
        }
    
    def example_inputs(self) -> List[Any]:
        """Provide example inputs"""
        return [
            {
                "image": np.zeros((512, 512)),
                "modality": "CT",
                "task": "analyze_point",
                "roi": {"x": 256, "y": 256, "radius": 10}
            }
        ]
    
    def example_outputs(self) -> List[Any]:
        """Provide example outputs"""
        return [
            {
                "modality": "CT",
                "point_analysis": {
                    "hu_value": -50.0,
                    "tissue_type": {"type": "fat", "icon": "🟡"},
                    "confidence": 0.95,
                    "reasoning": "HU value of -50.0 falls within fat range. Homogeneous region suggests uniform tissue"
                }
            }
        ]
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component

    
    def change(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        show_progress_on: Component | Sequence[Component] | None = None,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | Literal[True] | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, will use the functions name as the endpoint route. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            show_progress_on: Component or list of components to show the progress animation on. If None, will show the progress animation on all of the output components.
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
            key: A unique key for this event listener to be used in @gr.render(). If set, this value identifies an event as identical across re-renders when the key is identical.
        
        """
        ...
    
    def select(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        show_progress_on: Component | Sequence[Component] | None = None,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | Literal[True] | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, will use the functions name as the endpoint route. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            show_progress_on: Component or list of components to show the progress animation on. If None, will show the progress animation on all of the output components.
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
            key: A unique key for this event listener to be used in @gr.render(). If set, this value identifies an event as identical across re-renders when the key is identical.
        
        """
        ...
    
    def upload(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        show_progress_on: Component | Sequence[Component] | None = None,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | Literal[True] | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, will use the functions name as the endpoint route. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            show_progress_on: Component or list of components to show the progress animation on. If None, will show the progress animation on all of the output components.
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
            key: A unique key for this event listener to be used in @gr.render(). If set, this value identifies an event as identical across re-renders when the key is identical.
        
        """
        ...
    
    def clear(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        show_progress_on: Component | Sequence[Component] | None = None,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | Literal[True] | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, will use the functions name as the endpoint route. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            show_progress_on: Component or list of components to show the progress animation on. If None, will show the progress animation on all of the output components.
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps as well as the Clients to use this event. If fn is None, show_api will automatically be set to False.
            key: A unique key for this event listener to be used in @gr.render(). If set, this value identifies an event as identical across re-renders when the key is identical.
        
        """
        ...