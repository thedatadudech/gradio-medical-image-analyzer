#!/bin/bash
# Build script for Medical Image Analyzer frontend

echo "🏥 Building Medical Image Analyzer frontend..."

# Navigate to frontend directory
cd frontend

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the component
echo "🔨 Building component..."
npm run build

# Copy built files to templates
echo "📋 Copying built files to templates..."
mkdir -p ../backend/gradio_medical_image_analyzer/templates
cp -r dist/* ../backend/gradio_medical_image_analyzer/templates/

echo "✅ Frontend build complete!"