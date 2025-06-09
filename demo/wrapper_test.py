#!/usr/bin/env python3
"""
Frontend Wrapper Test für Medical Image Analyzer
Nutzt Standard Gradio Komponenten um die Backend-Funktionalität zu testen
"""

import gradio as gr
import numpy as np
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend'))

from gradio_medical_image_analyzer import MedicalImageAnalyzer
import cv2

def draw_roi_on_image(image, roi_x, roi_y, roi_radius):
    """Draw ROI circle on the image"""
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Draw ROI circle
    center = (int(roi_x), int(roi_y))
    radius = int(roi_radius)
    
    # Draw outer circle (white)
    cv2.circle(image_rgb, center, radius, (255, 255, 255), 2)
    # Draw inner circle (red)
    cv2.circle(image_rgb, center, radius-1, (255, 0, 0), 2)
    # Draw center cross
    cv2.line(image_rgb, (center[0]-5, center[1]), (center[0]+5, center[1]), (255, 0, 0), 2)
    cv2.line(image_rgb, (center[0], center[1]-5), (center[0], center[1]+5), (255, 0, 0), 2)
    
    return image_rgb

def create_fat_overlay(base_image, segmentation_results):
    """Create overlay image with fat segmentation highlighted"""
    # Convert to RGB
    if len(base_image.shape) == 2:
        overlay_img = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    else:
        overlay_img = base_image.copy()
    
    # Check if we have segmentation masks
    if not segmentation_results or 'segments' not in segmentation_results:
        return overlay_img
    
    segments = segmentation_results.get('segments', {})
    
    # Apply subcutaneous fat overlay (yellow)
    if 'subcutaneous' in segments and segments['subcutaneous'].get('mask') is not None:
        mask = segments['subcutaneous']['mask']
        yellow_overlay = np.zeros_like(overlay_img)
        yellow_overlay[mask > 0] = [255, 255, 0]  # Yellow
        overlay_img = cv2.addWeighted(overlay_img, 0.7, yellow_overlay, 0.3, 0)
    
    # Apply visceral fat overlay (red)
    if 'visceral' in segments and segments['visceral'].get('mask') is not None:
        mask = segments['visceral']['mask']
        red_overlay = np.zeros_like(overlay_img)
        red_overlay[mask > 0] = [255, 0, 0]  # Red
        overlay_img = cv2.addWeighted(overlay_img, 0.7, red_overlay, 0.3, 0)
    
    # Add legend
    cv2.putText(overlay_img, "Yellow: Subcutaneous Fat", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(overlay_img, "Red: Visceral Fat", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return overlay_img

def process_and_analyze(file_obj, task, roi_x, roi_y, roi_radius, symptoms, show_overlay=False):
    """
    Processes uploaded file and performs analysis
    """
    if file_obj is None:
        return None, "No file selected", "", {}, None
    
    # Create analyzer instance
    analyzer = MedicalImageAnalyzer(
        analysis_mode="structured",
        include_confidence=True,
        include_reasoning=True
    )
    
    try:
        # Process the file (DICOM or image)
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        pixel_array, display_array, metadata = analyzer.process_file(file_path)
        
        # Update modality from file metadata
        modality = metadata.get('modality', 'CR')
        
        # Prepare analysis parameters
        analysis_params = {
            "image": pixel_array,
            "modality": modality,
            "task": task
        }
        
        # Add ROI if applicable
        if task in ["analyze_point", "full_analysis"]:
            # Scale ROI coordinates to image size
            h, w = pixel_array.shape
            roi_x_scaled = int(roi_x * w / 512)  # Assuming slider max is 512
            roi_y_scaled = int(roi_y * h / 512)
            
            analysis_params["roi"] = {
                "x": roi_x_scaled,
                "y": roi_y_scaled,
                "radius": roi_radius
            }
        
        # Add clinical context
        if symptoms:
            analysis_params["clinical_context"] = ", ".join(symptoms)
        
        # Perform analysis
        results = analyzer.analyze_image(**analysis_params)
        
        # Create visual report
        if analyzer.analysis_mode == "visual":
            visual_report = analyzer._create_html_report(results)
        else:
            # Create our own visual report
            visual_report = create_visual_report(results, metadata)
        
        # Add metadata info
        info = f"📄 {metadata.get('file_type', 'Unknown')} | "
        info += f"🏥 {modality} | "
        info += f"📐 {metadata.get('shape', 'Unknown')}"
        
        if metadata.get('window_center'):
            info += f" | Window C:{metadata['window_center']:.0f} W:{metadata['window_width']:.0f}"
        
        # Create overlay image if requested
        overlay_image = None
        if show_overlay:
            # For ROI visualization
            if task in ["analyze_point", "full_analysis"] and roi_x and roi_y:
                overlay_image = draw_roi_on_image(display_array.copy(), roi_x, roi_y, roi_radius)
            
            # For fat segmentation overlay
            if task == "segment_fat" and 'segmentation' in results and modality == 'CT':
                # Get segmentation masks from results
                seg_results = {
                    'segments': {
                        'subcutaneous': {'mask': None},
                        'visceral': {'mask': None}
                    }
                }
                
                # Check if we have actual mask data
                if 'segments' in results['segmentation']:
                    seg_results = results['segmentation']
                
                overlay_image = create_fat_overlay(display_array.copy(), seg_results)
        
        return display_array, info, visual_report, results, overlay_image
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, error_msg, f"<div style='color: red;'>❌ {error_msg}</div>", {"error": error_msg}, None


def create_visual_report(results, metadata):
    """Creates a visual HTML report with improved styling"""
    html = f"""
    <div class='medical-report' style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
                padding: 24px; 
                background: #ffffff; 
                border-radius: 12px; 
                max-width: 100%; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                color: #1a1a1a !important;'>
        
        <h2 style='color: #1e40af !important; 
                   border-bottom: 3px solid #3b82f6; 
                   padding-bottom: 12px; 
                   margin-bottom: 20px;
                   font-size: 24px;
                   font-weight: 600;'>
            🏥 Medical Image Analysis Report
        </h2>
        
        <div style='background: #f0f9ff; 
                    padding: 20px; 
                    margin: 16px 0; 
                    border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <h3 style='color: #1e3a8a !important; 
                       font-size: 18px; 
                       font-weight: 600; 
                       margin-bottom: 12px;'>
                📋 Metadata
            </h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important; width: 40%;'><strong style='color: #374151 !important;'>File Type:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>{metadata.get('file_type', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>Modality:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>{results.get('modality', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>Image Size:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>{metadata.get('shape', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>Timestamp:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>{results.get('timestamp', 'N/A')}</td>
                </tr>
            </table>
        </div>
    """
    
    # Point Analysis
    if 'point_analysis' in results:
        pa = results['point_analysis']
        tissue = pa.get('tissue_type', {})
        
        html += f"""
        <div style='background: #f0f9ff; 
                    padding: 20px; 
                    margin: 16px 0; 
                    border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <h3 style='color: #1e3a8a !important; 
                       font-size: 18px; 
                       font-weight: 600; 
                       margin-bottom: 12px;'>
                🎯 Point Analysis
            </h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important; width: 40%;'><strong style='color: #374151 !important;'>Position:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>({pa.get('location', {}).get('x', 'N/A')}, {pa.get('location', {}).get('y', 'N/A')})</td>
                </tr>
        """
        
        if results.get('modality') == 'CT':
            html += f"""
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>HU Value:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important; font-weight: 500;'>{pa.get('hu_value', 'N/A'):.1f}</td>
                </tr>
            """
        else:
            html += f"""
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>Intensity:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>{pa.get('intensity', 'N/A'):.3f}</td>
                </tr>
            """
        
        html += f"""
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>Tissue Type:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>
                        <span style='font-size: 1.3em; vertical-align: middle;'>{tissue.get('icon', '')}</span> 
                        <span style='font-weight: 500; text-transform: capitalize;'>{tissue.get('type', 'Unknown').replace('_', ' ')}</span>
                    </td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; color: #4b5563 !important;'><strong style='color: #374151 !important;'>Confidence:</strong></td>
                    <td style='padding: 8px 0; color: #1f2937 !important;'>{pa.get('confidence', 'N/A')}</td>
                </tr>
            </table>
        """
        
        if 'reasoning' in pa:
            html += f"""
            <div style='margin-top: 12px; 
                        padding: 12px; 
                        background: #f0f7ff; 
                        border-left: 3px solid #0066cc; 
                        border-radius: 4px;'>
                <p style='margin: 0; color: #4b5563 !important; font-style: italic;'>
                    💭 {pa['reasoning']}
                </p>
            </div>
            """
        
        html += "</div>"
    
    # Segmentation Results
    if 'segmentation' in results and results['segmentation']:
        seg = results['segmentation']
        
        if 'statistics' in seg:
            # Fat segmentation for CT
            stats = seg['statistics']
            html += f"""
            <div style='background: #f0f9ff; 
                        padding: 20px; 
                        margin: 16px 0; 
                        border-radius: 8px; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <h3 style='color: #1e3a8a !important; 
                           font-size: 18px; 
                           font-weight: 600; 
                           margin-bottom: 12px;'>
                    🔬 Fat Segmentation Analysis
                </h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px;'>
                    <div style='padding: 16px; background: #ffffff; border-radius: 6px; border: 1px solid #e5e7eb;'>
                        <h4 style='color: #6b7280 !important; font-size: 14px; margin: 0 0 8px 0; font-weight: 500;'>Total Fat</h4>
                        <p style='color: #1f2937 !important; font-size: 24px; font-weight: 600; margin: 0;'>{stats.get('total_fat_percentage', 0):.1f}%</p>
                    </div>
                    <div style='padding: 16px; background: #fffbeb; border-radius: 6px; border: 1px solid #fbbf24;'>
                        <h4 style='color: #92400e !important; font-size: 14px; margin: 0 0 8px 0; font-weight: 500;'>Subcutaneous</h4>
                        <p style='color: #d97706 !important; font-size: 24px; font-weight: 600; margin: 0;'>{stats.get('subcutaneous_fat_percentage', 0):.1f}%</p>
                    </div>
                    <div style='padding: 16px; background: #fef2f2; border-radius: 6px; border: 1px solid #fca5a5;'>
                        <h4 style='color: #991b1b !important; font-size: 14px; margin: 0 0 8px 0; font-weight: 500;'>Visceral</h4>
                        <p style='color: #dc2626 !important; font-size: 24px; font-weight: 600; margin: 0;'>{stats.get('visceral_fat_percentage', 0):.1f}%</p>
                    </div>
                    <div style='padding: 16px; background: #eff6ff; border-radius: 6px; border: 1px solid #93c5fd;'>
                        <h4 style='color: #1e3a8a !important; font-size: 14px; margin: 0 0 8px 0; font-weight: 500;'>V/S Ratio</h4>
                        <p style='color: #1e40af !important; font-size: 24px; font-weight: 600; margin: 0;'>{stats.get('visceral_subcutaneous_ratio', 0):.2f}</p>
                    </div>
                </div>
            """
            
            if 'interpretation' in seg:
                interp = seg['interpretation']
                obesity_color = "#27ae60" if interp.get("obesity_risk") == "normal" else "#f39c12" if interp.get("obesity_risk") == "moderate" else "#e74c3c"
                visceral_color = "#27ae60" if interp.get("visceral_risk") == "normal" else "#f39c12" if interp.get("visceral_risk") == "moderate" else "#e74c3c"
                
                html += f"""
                <div style='margin-top: 16px; padding: 16px; background: #f8f9fa; border-radius: 6px;'>
                    <h4 style='color: #333; font-size: 16px; font-weight: 600; margin-bottom: 8px;'>Risk Assessment</h4>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
                        <div>
                            <span style='color: #666; font-size: 14px;'>Obesity Risk:</span>
                            <span style='color: {obesity_color}; font-weight: 600; margin-left: 8px;'>{interp.get('obesity_risk', 'N/A').upper()}</span>
                        </div>
                        <div>
                            <span style='color: #666; font-size: 14px;'>Visceral Risk:</span>
                            <span style='color: {visceral_color}; font-weight: 600; margin-left: 8px;'>{interp.get('visceral_risk', 'N/A').upper()}</span>
                        </div>
                    </div>
                """
                
                if interp.get('recommendations'):
                    html += """
                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #e0e0e0;'>
                        <h5 style='color: #333; font-size: 14px; font-weight: 600; margin-bottom: 8px;'>💡 Recommendations</h5>
                        <ul style='margin: 0; padding-left: 20px; color: #555;'>
                    """
                    for rec in interp['recommendations']:
                        html += f"<li style='margin: 4px 0;'>{rec}</li>"
                    html += "</ul></div>"
                
                html += "</div>"
            html += "</div>"
        
        elif 'tissue_distribution' in seg:
            # X-ray tissue distribution
            html += f"""
            <div style='background: white; 
                        padding: 20px; 
                        margin: 16px 0; 
                        border-radius: 8px; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <h3 style='color: #333; 
                           font-size: 18px; 
                           font-weight: 600; 
                           margin-bottom: 12px;'>
                    🦴 Tissue Distribution
                </h3>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px;'>
            """
            
            tissue_dist = seg['tissue_distribution']
            tissue_icons = {
                'bone': '🦴', 'soft_tissue': '🔴', 'air': '🌫️',
                'metal': '⚙️', 'fat': '🟡', 'fluid': '💧'
            }
            
            tissue_colors = {
                'bone': '#fff7e6',
                'soft_tissue': '#fee',
                'air': '#e6f3ff',
                'metal': '#f0f0f0',
                'fat': '#fffbe6',
                'fluid': '#e6f7ff'
            }
            
            for tissue, percentage in tissue_dist.items():
                if percentage > 0:
                    icon = tissue_icons.get(tissue, '📍')
                    bg_color = tissue_colors.get(tissue, '#f8f9fa')
                    html += f"""
                    <div style='text-align: center; 
                                padding: 16px; 
                                background: {bg_color}; 
                                border-radius: 8px;
                                transition: transform 0.2s;'>
                        <div style='font-size: 2.5em; margin-bottom: 8px;'>{icon}</div>
                        <div style='font-weight: 600; 
                                    color: #333; 
                                    font-size: 14px; 
                                    margin-bottom: 4px;'>
                            {tissue.replace('_', ' ').title()}
                        </div>
                        <div style='color: #0066cc; 
                                    font-size: 20px; 
                                    font-weight: 700;'>
                            {percentage:.1f}%
                        </div>
                    </div>
                    """
            
            html += "</div>"
            
            if seg.get('clinical_findings'):
                html += """
                <div style='margin-top: 16px; 
                            padding: 16px; 
                            background: #fff3cd; 
                            border-left: 4px solid #ffc107; 
                            border-radius: 4px;'>
                    <h4 style='color: #856404; 
                               font-size: 16px; 
                               font-weight: 600; 
                               margin: 0 0 8px 0;'>
                        ⚠️ Clinical Findings
                    </h4>
                    <ul style='margin: 0; padding-left: 20px; color: #856404;'>
                """
                for finding in seg['clinical_findings']:
                    html += f"<li style='margin: 4px 0;'>{finding.get('description', 'Unknown finding')}</li>"
                html += "</ul></div>"
            
            html += "</div>"
    
    # Quality Assessment
    if 'quality_metrics' in results:
        quality = results['quality_metrics']
        quality_colors = {
            'excellent': '#27ae60',
            'good': '#27ae60',
            'fair': '#f39c12',
            'poor': '#e74c3c',
            'unknown': '#95a5a6'
        }
        q_color = quality_colors.get(quality.get('overall_quality', 'unknown'), '#95a5a6')
        
        html += f"""
        <div style='background: #f0f9ff; 
                    padding: 20px; 
                    margin: 16px 0; 
                    border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <h3 style='color: #1e3a8a !important; 
                       font-size: 18px; 
                       font-weight: 600; 
                       margin-bottom: 12px;'>
                📊 Image Quality Assessment
            </h3>
            <div style='display: flex; align-items: center; gap: 16px;'>
                <div>
                    <span style='color: #4b5563 !important; font-size: 14px;'>Overall Quality:</span>
                    <span style='color: {q_color} !important; 
                                 font-size: 18px; 
                                 font-weight: 700; 
                                 margin-left: 8px;'>
                        {quality.get('overall_quality', 'unknown').upper()}
                    </span>
                </div>
            </div>
        """
        
        if quality.get('issues'):
            html += f"""
            <div style='margin-top: 12px; 
                        padding: 12px; 
                        background: #fff3cd; 
                        border-left: 3px solid #ffc107; 
                        border-radius: 4px;'>
                <strong style='color: #856404;'>Issues Detected:</strong>
                <ul style='margin: 4px 0 0 0; padding-left: 20px; color: #856404;'>
            """
            for issue in quality['issues']:
                html += f"<li style='margin: 2px 0;'>{issue}</li>"
            html += "</ul></div>"
        
        html += "</div>"
    
    html += "</div>"
    return html


# Create Gradio interface
with gr.Blocks(
    title="Medical Image Analyzer - Wrapper Test", 
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        text_size="md",
        spacing_size="md",
        radius_size="md",
    ).set(
        # Medical blue theme colors
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        border_color_primary="*primary_600",
        border_color_primary_dark="*primary_600",
        # Text colors for better contrast
        body_text_color="*neutral_100",
        body_text_color_dark="*neutral_100",
        body_text_color_subdued="*neutral_300",
        body_text_color_subdued_dark="*neutral_300",
        # Button colors
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_dark="*primary_600",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
    ),
    css="""
    /* Medical blue theme with high contrast */
    :root {
        --medical-blue: #1e40af;
        --medical-blue-light: #3b82f6;
        --medical-blue-dark: #1e3a8a;
        --text-primary: #f9fafb;
        --text-secondary: #e5e7eb;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
    }
    
    /* Override default text colors for medical theme */
    * {
        color: var(--text-primary) !important;
    }
    
    /* Fix text contrast issues */
    .svelte-12ioyct { color: var(--text-primary) !important; }
    
    /* Override table cell colors for better readability */
    td { 
        color: var(--text-primary) !important; 
        padding: 8px 12px !important;
    }
    td strong { 
        color: var(--text-primary) !important; 
        font-weight: 600 !important; 
    }
    
    /* Fix upload component text and translate */
    .wrap.svelte-12ioyct::before {
        content: 'Drop file here' !important;
        color: var(--text-primary) !important;
    }
    .wrap.svelte-12ioyct::after {
        content: 'Click to upload' !important;
        color: var(--text-secondary) !important;
    }
    .wrap.svelte-12ioyct span { 
        display: none !important; /* Hide German text */
    }
    
    /* Style the file upload area */
    .file-upload {
        border: 2px dashed var(--medical-blue-light) !important;
        border-radius: 8px !important;
        padding: 20px !important;
        text-align: center !important;
        background: var(--bg-secondary) !important;
        transition: all 0.3s ease !important;
        color: var(--text-primary) !important;
    }
    
    .file-upload:hover {
        border-color: var(--medical-blue) !important;
        background: var(--bg-tertiary) !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Make sure all text in tables is readable */
    table { 
        width: 100%;
        border-collapse: collapse;
    }
    th { 
        font-weight: 600; 
        background: var(--bg-tertiary);
        padding: 8px 12px;
    }
    
    /* Ensure report text is readable with white background */
    .medical-report {
        background: #ffffff !important;
        border: 2px solid var(--medical-blue-light) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        color: #1a1a1a !important;
    }
    
    .medical-report * {
        color: #1f2937 !important; /* Dark gray text */
    }
    
    .medical-report h2 {
        color: #1e40af !important; /* Medical blue for main heading */
    }
    
    .medical-report h3, .medical-report h4 {
        color: #1e3a8a !important; /* Darker medical blue for subheadings */
    }
    
    .medical-report strong {
        color: #374151 !important; /* Darker gray for labels */
    }
    
    .medical-report td {
        color: #1f2937 !important; /* Ensure table text is dark */
    }
    
    /* Report sections with light blue background */
    .medical-report > div {
        background: #f0f9ff !important;
        color: #1f2937 !important;
    }
    
    /* Medical blue accents for UI elements */
    .gr-button-primary {
        background: var(--medical-blue) !important;
        border-color: var(--medical-blue) !important;
    }
    
    .gr-button-primary:hover {
        background: var(--medical-blue-dark) !important;
        border-color: var(--medical-blue-dark) !important;
    }
    
    /* Accordion styling with medical theme */
    .gr-accordion {
        border-color: var(--medical-blue-light) !important;
    }
    
    /* Slider track in medical blue */
    input[type="range"]::-webkit-slider-track {
        background: var(--bg-tertiary) !important;
    }
    
    input[type="range"]::-webkit-slider-thumb {
        background: var(--medical-blue) !important;
    }
    
    /* Tab styling */
    .gr-tab-item {
        border-color: var(--medical-blue-light) !important;
    }
    
    .gr-tab-item.selected {
        background: var(--medical-blue) !important;
        color: white !important;
    }
    """
) as demo:
    gr.Markdown("""
    # 🏥 Medical Image Analyzer
    
    Supports **DICOM** (.dcm) and all image formats with automatic modality detection!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload with custom styling - no file type restrictions
            with gr.Group():
                gr.Markdown("### 📤 Upload Medical Image")
                file_input = gr.File(
                    label="Select Medical Image File (.dcm, .dicom, IM_*, .png, .jpg, etc.)",
                    file_count="single",
                    type="filepath",
                    elem_classes="file-upload"
                    # Note: NO file_types parameter = accepts ALL files
                )
                gr.Markdown("""
                <small style='color: #666;'>
                Accepts: DICOM (.dcm, .dicom), Images (.png, .jpg, .jpeg, .tiff, .bmp), 
                and files without extensions (e.g., IM_0001, IM_0002, etc.)
                </small>
                """)
            
            # Task selection
            task = gr.Dropdown(
                choices=[
                    ("🎯 Point Analysis", "analyze_point"),
                    ("🔬 Fat Segmentation (CT only)", "segment_fat"),
                    ("📊 Full Analysis", "full_analysis")
                ],
                value="full_analysis",
                label="Analysis Task"
            )
            
            # ROI settings
            with gr.Accordion("🎯 Region of Interest (ROI)", open=True):
                roi_x = gr.Slider(0, 512, 256, label="X Position", step=1)
                roi_y = gr.Slider(0, 512, 256, label="Y Position", step=1)
                roi_radius = gr.Slider(5, 50, 10, label="Radius", step=1)
            
            # Clinical context
            with gr.Accordion("🏥 Clinical Context", open=False):
                symptoms = gr.CheckboxGroup(
                    choices=[
                        "Dyspnea", "Chest Pain", "Abdominal Pain",
                        "Trauma", "Obesity Screening", "Routine Check"
                    ],
                    label="Symptoms/Indication"
                )
            
            # Visualization options
            with gr.Accordion("🎨 Visualization Options", open=True):
                show_overlay = gr.Checkbox(
                    label="Show ROI/Segmentation Overlay",
                    value=True,
                    info="Display ROI circle or fat segmentation overlay on the image"
                )
            
            analyze_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            # Results with tabs for different views
            with gr.Tab("🖼️ Original Image"):
                image_display = gr.Image(label="Medical Image", type="numpy")
                
            with gr.Tab("🎯 Overlay View"):
                overlay_display = gr.Image(label="Image with Overlay", type="numpy")
            
            file_info = gr.Textbox(label="File Information", lines=1)
            
            with gr.Tab("📊 Visual Report"):
                report_html = gr.HTML()
            
            with gr.Tab("🔧 JSON Output"):
                json_output = gr.JSON(label="Structured Data for AI Agents")
    
    # Examples and help
    with gr.Row():
        gr.Markdown("""
        ### 📁 Supported Formats
        - **DICOM**: Automatic HU value extraction and modality detection
        - **PNG/JPG**: Interpreted as X-ray (CR) unless filename contains 'CT'
        - **All Formats**: Automatic grayscale conversion
        - **Files without extension**: Supported (e.g., IM_0001)
        
        ### 🎯 Usage
        1. Upload a medical image file
        2. Select the desired analysis task
        3. For point analysis: Adjust the ROI position
        4. Click "Analyze"
        
        ### 💡 Tips
        - The ROI sliders will automatically adjust to your image size
        - CT images show HU values, X-rays show intensity values
        - Fat segmentation is only available for CT images
        """)
    
    # Connect the interface
    analyze_btn.click(
        fn=process_and_analyze,
        inputs=[file_input, task, roi_x, roi_y, roi_radius, symptoms, show_overlay],
        outputs=[image_display, file_info, report_html, json_output, overlay_display]
    )
    
    # Auto-update ROI limits when image is loaded
    def update_roi_on_upload(file_obj):
        if file_obj is None:
            return gr.update(), gr.update()
        
        try:
            analyzer = MedicalImageAnalyzer()
            _, _, metadata = analyzer.process_file(file_obj.name if hasattr(file_obj, 'name') else str(file_obj))
            
            if 'shape' in metadata:
                h, w = metadata['shape']
                return gr.update(maximum=w-1, value=w//2), gr.update(maximum=h-1, value=h//2)
        except:
            pass
        
        return gr.update(), gr.update()
    
    file_input.change(
        fn=update_roi_on_upload,
        inputs=[file_input],
        outputs=[roi_x, roi_y]
    )

    # Add custom JavaScript to translate upload text
    demo.load(
        None,
        None,
        None,
        js="""
        () => {
            // Wait for the page to load
            setTimeout(() => {
                // Find and replace German text in upload component
                const uploadElements = document.querySelectorAll('.wrap.svelte-12ioyct');
                uploadElements.forEach(el => {
                    if (el.textContent.includes('Datei hier ablegen')) {
                        el.innerHTML = el.innerHTML
                            .replace('Datei hier ablegen', 'Drop file here')
                            .replace('oder', 'or')
                            .replace('Hochladen', 'Click to upload');
                    }
                });
                
                // Also update the button text if it exists
                const uploadButtons = document.querySelectorAll('button');
                uploadButtons.forEach(btn => {
                    if (btn.textContent.includes('Hochladen')) {
                        btn.textContent = 'Upload';
                    }
                });
            }, 100);
        }
        """
    )

if __name__ == "__main__":
    demo.launch()