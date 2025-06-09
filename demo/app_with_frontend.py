#!/usr/bin/env python3
"""
Demo für MedicalImageAnalyzer mit Frontend Component
Zeigt die Verwendung der vollständigen Gradio Custom Component
"""

import gradio as gr
import numpy as np
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend'))

from gradio_medical_image_analyzer import MedicalImageAnalyzer

# Example data for demos
EXAMPLE_DATA = [
    {
        "image": {"url": "examples/ct_chest.png"},
        "analysis": {
            "modality": "CT",
            "point_analysis": {
                "tissue_type": {"icon": "🟡", "type": "fat"},
                "hu_value": -75.0
            },
            "segmentation": {
                "interpretation": {
                    "obesity_risk": "moderate"
                }
            }
        }
    },
    {
        "image": {"url": "examples/xray_chest.png"},
        "analysis": {
            "modality": "CR",
            "point_analysis": {
                "tissue_type": {"icon": "🦴", "type": "bone"}
            }
        }
    }
]

def create_demo():
    with gr.Blocks(title="Medical Image Analyzer - Component Demo") as demo:
        gr.Markdown("""
        # 🏥 Medical Image Analyzer - Frontend Component Demo
        
        Diese Demo zeigt die vollständige Gradio Custom Component mit Frontend-Integration.
        Unterstützt DICOM-Dateien und alle gängigen Bildformate.
        """)
        
        with gr.Row():
            with gr.Column():
                # Configuration
                gr.Markdown("### ⚙️ Konfiguration")
                
                analysis_mode = gr.Radio(
                    choices=["structured", "visual"],
                    value="structured",
                    label="Analyse-Modus",
                    info="structured: für AI Agents, visual: für Menschen"
                )
                
                include_confidence = gr.Checkbox(
                    value=True,
                    label="Konfidenzwerte einschließen"
                )
                
                include_reasoning = gr.Checkbox(
                    value=True,
                    label="Reasoning einschließen"
                )
                
            with gr.Column(scale=2):
                # The custom component
                analyzer = MedicalImageAnalyzer(
                    label="Medical Image Analyzer",
                    analysis_mode="structured",
                    include_confidence=True,
                    include_reasoning=True,
                    elem_id="medical-analyzer"
                )
        
        # Examples section
        gr.Markdown("### 📁 Beispiele")
        
        examples = gr.Examples(
            examples=EXAMPLE_DATA,
            inputs=analyzer,
            label="Beispiel-Analysen"
        )
        
        # Info section
        gr.Markdown("""
        ### 📝 Verwendung
        
        1. **Datei hochladen**: Ziehen Sie eine DICOM- oder Bilddatei in den Upload-Bereich
        2. **Modalität wählen**: CT, CR, DX, RX, oder DR
        3. **Analyse-Task**: Punktanalyse, Fettsegmentierung, oder vollständige Analyse
        4. **ROI aktivieren**: Klicken Sie auf das Bild, um einen Analysepunkt zu wählen
        
        ### 🔧 Features
        
        - **DICOM Support**: Automatische Erkennung von Modalität und HU-Werten
        - **Multi-Tissue Segmentation**: Erkennt Knochen, Weichgewebe, Luft, Metall, Fett, Flüssigkeit
        - **Klinische Bewertung**: Adipositas-Risiko, Gewebeverteilung, Anomalieerkennung
        - **AI-Agent Ready**: Strukturierte JSON-Ausgabe für Integration
        
        ### 🔗 Integration
        
        ```python
        import gradio as gr
        from gradio_medical_image_analyzer import MedicalImageAnalyzer
        
        analyzer = MedicalImageAnalyzer(
            analysis_mode="structured",
            include_confidence=True
        )
        
        # Use in your Gradio app
        with gr.Blocks() as app:
            analyzer_component = analyzer
            # ... rest of your app
        ```
        """)
        
        # Event handlers
        def update_config(mode, conf, reason):
            # This would update the component configuration
            # In real implementation, this would be handled by the component
            return gr.update(
                analysis_mode=mode,
                include_confidence=conf,
                include_reasoning=reason
            )
        
        # Connect configuration changes
        for config in [analysis_mode, include_confidence, include_reasoning]:
            config.change(
                fn=update_config,
                inputs=[analysis_mode, include_confidence, include_reasoning],
                outputs=analyzer
            )
        
        # Handle analysis results
        def handle_analysis_complete(data):
            if data and "analysis" in data:
                analysis = data["analysis"]
                report = data.get("report", "")
                
                # Log to console for debugging
                print("Analysis completed:")
                print(f"Modality: {analysis.get('modality', 'Unknown')}")
                if "point_analysis" in analysis:
                    print(f"Tissue: {analysis['point_analysis'].get('tissue_type', {}).get('type', 'Unknown')}")
                
                return data
            return data
        
        analyzer.change(
            fn=handle_analysis_complete,
            inputs=analyzer,
            outputs=analyzer
        )
    
    return demo


def create_simple_demo():
    """Einfache Demo ohne viel Konfiguration"""
    with gr.Blocks(title="Medical Image Analyzer - Simple Demo") as demo:
        gr.Markdown("# 🏥 Medical Image Analyzer")
        
        analyzer = MedicalImageAnalyzer(
            label="Laden Sie ein medizinisches Bild hoch (DICOM, PNG, JPG)",
            analysis_mode="visual",  # Visual mode for human-readable output
            elem_id="analyzer"
        )
        
        # Auto-analyze on upload
        @analyzer.upload
        def auto_analyze(file_data):
            # The component handles the analysis internally
            return file_data
        
    return demo


if __name__ == "__main__":
    # You can switch between demos
    # demo = create_demo()  # Full demo with configuration
    demo = create_simple_demo()  # Simple demo
    
    demo.launch()