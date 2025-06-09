#!/usr/bin/env python3
"""
Test script for Medical Image Analyzer integration
"""

import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from gradio_medical_image_analyzer import MedicalImageAnalyzer

def test_ct_analysis():
    """Test CT image analysis with fat segmentation"""
    print("Testing CT Analysis...")
    
    # Create a simulated CT image with different tissue types
    ct_image = np.zeros((512, 512), dtype=np.float32)
    
    # Add different tissue regions based on HU values
    # Air region (-1000 HU)
    ct_image[50:150, 50:150] = -1000
    
    # Fat region (-75 HU)
    ct_image[200:300, 200:300] = -75
    
    # Soft tissue region (40 HU)
    ct_image[350:450, 350:450] = 40
    
    # Bone region (300 HU)
    ct_image[100:200, 350:450] = 300
    
    # Create analyzer
    analyzer = MedicalImageAnalyzer(
        analysis_mode="structured",
        include_confidence=True,
        include_reasoning=True
    )
    
    # Test point analysis
    print("\n1. Testing point analysis...")
    result = analyzer.process_image(
        image=ct_image,
        modality="CT",
        task="analyze_point",
        roi={"x": 250, "y": 250, "radius": 10}
    )
    
    print(f"   HU Value: {result.get('point_analysis', {}).get('hu_value', 'N/A')}")
    print(f"   Tissue Type: {result.get('point_analysis', {}).get('tissue_type', {}).get('type', 'N/A')}")
    print(f"   Confidence: {result.get('point_analysis', {}).get('confidence', 'N/A')}")
    
    # Test fat segmentation
    print("\n2. Testing fat segmentation...")
    result = analyzer.process_image(
        image=ct_image,
        modality="CT",
        task="segment_fat"
    )
    
    if "error" in result.get("segmentation", {}):
        print(f"   Error: {result['segmentation']['error']}")
    else:
        stats = result.get("segmentation", {}).get("statistics", {})
        print(f"   Total Fat %: {stats.get('total_fat_percentage', 'N/A')}")
        print(f"   Subcutaneous Fat %: {stats.get('subcutaneous_fat_percentage', 'N/A')}")
        print(f"   Visceral Fat %: {stats.get('visceral_fat_percentage', 'N/A')}")
    
    # Test full analysis
    print("\n3. Testing full analysis...")
    result = analyzer.process_image(
        image=ct_image,
        modality="CT",
        task="full_analysis",
        clinical_context="Patient with obesity concerns"
    )
    
    print(f"   Modality: {result.get('modality', 'N/A')}")
    print(f"   Quality: {result.get('quality_metrics', {}).get('overall_quality', 'N/A')}")
    if "clinical_correlation" in result:
        print(f"   Clinical Note: {result['clinical_correlation'].get('summary', 'N/A')}")


def test_xray_analysis():
    """Test X-ray image analysis"""
    print("\n\nTesting X-Ray Analysis...")
    
    # Create a simulated X-ray image with different intensities
    xray_image = np.zeros((512, 512), dtype=np.float32)
    
    # Background soft tissue
    xray_image[:, :] = 0.4
    
    # Bone region (high intensity)
    xray_image[100:400, 200:250] = 0.8  # Femur-like structure
    
    # Air/lung region (low intensity)
    xray_image[50:200, 50:200] = 0.1
    xray_image[50:200, 312:462] = 0.1
    
    # Metal implant (very high intensity)
    xray_image[250:280, 220:230] = 0.95
    
    # Create analyzer
    analyzer = MedicalImageAnalyzer(
        analysis_mode="structured",
        include_confidence=True,
        include_reasoning=True
    )
    
    # Test full X-ray analysis
    print("\n1. Testing full X-ray analysis...")
    result = analyzer.process_image(
        image=xray_image,
        modality="X-Ray",
        task="full_analysis"
    )
    
    if "segmentation" in result:
        segments = result["segmentation"].get("segments", {})
        print("   Detected tissues:")
        for tissue, info in segments.items():
            if info.get("present", False):
                print(f"     - {tissue}: {info.get('percentage', 0):.1f}%")
        
        # Check for findings
        findings = result["segmentation"].get("clinical_findings", [])
        if findings:
            print("   Clinical findings:")
            for finding in findings:
                print(f"     - {finding.get('description', 'Unknown finding')}")
    
    # Test point analysis on X-ray
    print("\n2. Testing X-ray point analysis...")
    result = analyzer.process_image(
        image=xray_image,
        modality="X-Ray",
        task="analyze_point",
        roi={"x": 225, "y": 265, "radius": 5}  # Metal implant region
    )
    
    point_analysis = result.get("point_analysis", {})
    print(f"   Tissue Type: {point_analysis.get('tissue_type', {}).get('type', 'N/A')}")
    print(f"   Intensity: {point_analysis.get('intensity', 'N/A')}")


def test_error_handling():
    """Test error handling"""
    print("\n\nTesting Error Handling...")
    
    analyzer = MedicalImageAnalyzer()
    
    # Test with invalid image
    print("\n1. Testing with None image...")
    result = analyzer.process_image(
        image=None,
        modality="CT",
        task="full_analysis"
    )
    print(f"   Error handled: {'error' in result}")
    
    # Test with invalid modality
    print("\n2. Testing with invalid modality...")
    result = analyzer.process_image(
        image=np.zeros((100, 100)),
        modality="MRI",  # Not supported
        task="full_analysis"
    )
    print(f"   Processed as: {result.get('modality', 'Unknown')}")


if __name__ == "__main__":
    print("Medical Image Analyzer Integration Test")
    print("=" * 50)
    
    test_ct_analysis()
    test_xray_analysis()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("Integration test completed!")
    print("\nNote: This test uses simulated data.")
    print("For real medical images, results will be more accurate.")