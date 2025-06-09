
# Medical Image Analyzer Demo

This folder contains demo applications for the `gradio_medical_image_analyzer` custom component.


## ⚠️ IMPORTANT MEDICAL DISCLAIMER ⚠️

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

🚨 **DO NOT USE FOR CLINICAL DIAGNOSIS OR MEDICAL DECISION MAKING** 🚨

This component is in **EARLY DEVELOPMENT** and is intended as a **proof of concept** for medical image analysis integration with Gradio. The results produced by this software:

- **ARE NOT** validated for clinical use
- **ARE NOT** FDA approved or CE marked
- **SHOULD NOT** be used for patient diagnosis or treatment decisions
- **SHOULD NOT** replace professional medical judgment
- **MAY CONTAIN** significant errors or inaccuracies
- **ARE PROVIDED** without any warranty of accuracy or fitness for medical purposes

**ALWAYS CONSULT QUALIFIED HEALTHCARE PROFESSIONALS** for medical image interpretation and clinical decisions. This software is intended solely for:
- Research and development purposes
- Educational demonstrations
- Technical integration testing
- Non-clinical experimental use

By using this software, you acknowledge that you understand these limitations and agree not to use it for any clinical or medical diagnostic purposes.


## Demo Files

- **`app.py`** - Main demo application showcasing all features of the medical image analyzer
- **`space.py`** - Hugging Face Spaces-optimized version
- **`app_with_frontend.py`** - Demo with custom frontend integration
- **`wrapper_test.py`** - Test file for component wrapper functionality
- **`css.css`** - Custom styling for the demo interface

## Installation

```bash
pip install -r requirements.txt
```

## Running the Demo

### Local Development
```bash
python app.py
```

The demo will be available at `http://localhost:7860`

### Hugging Face Spaces
```bash
python space.py
```

## Features Demonstrated

1. **File Upload Support**
   - DICOM files (.dcm, .dicom)
   - Standard image formats (PNG, JPG, TIFF, BMP)
   - Files without extensions (e.g., IM_0001)

2. **Analysis Tasks**
   - 🎯 Point Analysis - Analyze specific ROI in the image
   - 🔬 Fat Segmentation - CT-specific fat tissue analysis
   - 📊 Full Analysis - Comprehensive image analysis

3. **Modality Support**
   - CT (Computed Tomography)
   - CR (Computed Radiography)
   - DX/RX/DR (Digital X-ray variants)

4. **Interactive Features**
   - ROI selection with sliders
   - Clinical context input
   - Overlay visualization
   - Real-time analysis

## Sample Workflow

1. Upload a medical image (DICOM or standard format)
2. Select the imaging modality (auto-detected for DICOM)
3. Choose an analysis task
4. Adjust ROI if needed
5. Click "Analyze" to get results

## Output Formats

- **Visual Report**: HTML-formatted analysis results
- **JSON Output**: Structured data for AI agents and integration
- **Overlay View**: Visual annotations on the original image

## Development Notes

- The demo uses a dark medical theme optimized for clinical environments
- All processing is done locally - no data is sent to external servers
- The component is designed for both human interpretation and AI agent integration

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that you have the correct Python version (3.8+)
3. For DICOM files, ensure they are valid medical images
4. Report issues at: https://github.com/thedatadudech/gradio-medical-image-analyzer/issues

---

Developed for veterinary medicine with ❤️ and cutting-edge web technology

**Gradio Agents & MCP Hackathon 2025 - Track 2 Submission**
