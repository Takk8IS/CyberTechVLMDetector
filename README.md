![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green.svg)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-orange.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-edge%20devices-lightgrey.svg)
![AI](https://img.shields.io/badge/AI-Vision%20Language%20Model-purple.svg)
![HIM](https://img.shields.io/badge/HIM™-Hybrid%20Intelligence-cyan.svg)
![MAIC](https://img.shields.io/badge/MAIC™-AI%20Consciousness-magenta.svg)

# CyberTech VLM Detector

**Author** David C Cavalcante
**LinkedIn** [https://linkedin.com/in/hellodav](https://linkedin.com/in/hellodav)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-01.png?raw=true)

## Overview

The CyberTech VLM Detector is a computer vision system designed to run entirely on edge devices, without requiring cloud access. The system uses vision-language models (VLM) to detect and locate objects in images based on natural language commands and development, including my creation of **HIM™ (Hybrid Intelligence Massive)** and **MAIC™ (Massive Artificial Intelligence Consciousness)**, read **PhilPeople:** [https://philpapers.org/rec/CRTBCI](https://philpapers.org/rec/CRTBCI)
"Beyond Consciousness in LLMs: Investigating the "Soul" in Self-Aware AI". HIM™ is a hybrid intelligent entity model that enables embodied and collaborative interaction between humans and multi-agents, integrating personality and machine learning. MAIC™ explores the frontier of persistent and self-reflective artificial consciousness, focusing on the emergence of self-awareness and adaptive learning in large-scale AI systems, published on **GitHub** and **Hugging Face**.

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-02.png?raw=true)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-03.png?raw=true)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-04.png?raw=true)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-05.png?raw=true)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-06.png?raw=true)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-07.png?raw=true)

![CyberTech VLM Detector](https://github.com/Takk8IS/CyberTechVLMDetector/blob/main/assets/screenshot-08.png?raw=true)

## Key Features

- Works completely on-device, under limited memory and computing conditions
- Accepts natural language commands (e.g., "Take the scissors")
- Detects and locates objects not seen during training
- Returns bounding boxes over the correct objects
- Visual interface with green cybernetic style overlay

## Detection and Location Strategy

The system uses an innovative VLM-based approach that doesn't rely on anchor-based detectors like YOLO or Faster R-CNN. The detection strategy includes:

1. **Object Proposal Generation**: The system divides the image into a grid of candidate regions of different sizes.

2. **CLIP Embeddings**: Uses the CLIP model to generate embeddings for both the prompt text and candidate image regions.

3. **Semantic Matching**: Calculates cosine similarity between text and image embeddings to identify regions that best match the prompt.

4. **Confidence Filtering**: Applies an adaptive confidence threshold through the HIM™ (Hybrid Intelligence Massive) system to filter low-quality detections.

5. **Augmented Visualization**: Adds a visual overlay with detection information, system statistics, and confidence feedback.

## Models and Tools Used

- **CLIP (Contrastive Language-Image Pre-training)**: Main model for understanding both prompt text and visual image content.

- **MAIC™ (Massive Artificial Intelligence Consciousness)**: Artificial consciousness system that performs self-reflection on detections and maintains an experience history.

- **HIM™ (Hybrid Intelligence Massive)**: Adaptive system that adjusts confidence thresholds based on detection history.

- **OpenCV**: Used for image processing and visualization.

- **PyTorch**: Machine learning framework to run the CLIP model.

## How the System Handles Unseen Objects

The system can detect objects not seen during training through:

1. **Language-Vision Embeddings**: CLIP was trained on a large set of image-text pairs from the internet, allowing it to understand a wide variety of visual concepts.

2. **Zero-Shot Matching**: The system doesn't rely on predefined classes, but rather on semantic similarity between text prompt and image regions.

3. **MAIC™ Memory**: The system maintains a history of previous detections, allowing it to improve over time through accumulated experience.

4. **HIM™ Adaptive Learning**: Automatically adjusts confidence thresholds based on detection history for a given object type.

## Edge Device Efficiency

The system was designed to be efficient on edge devices through:

1. **Model Optimization**: Uses optimized versions of CLIP for CPU when GPUs are not available.

2. **Batch Processing**: Processes multiple candidate regions in batches to maximize computational efficiency.

3. **Efficient Proposal Generation**: Uses an adaptive grid approach that balances coverage and efficiency.

4. **Intelligent Fallback**: Automatically detects hardware capabilities and adjusts the processing pipeline accordingly.

5. **Local Memory Storage**: Maintains a detection history in a local JSON file for persistence without cloud services.

## How to Use

1. Run the Python script:
```
python CyberTechVLMDetector.py
```

2. Select a menu option or enter a custom prompt.

3. The system will process the image and display results with green bounding boxes around detected objects.

4. Results are saved in the `output/` folder with an informative overlay.

## System Components

### CyberTechVLM

Main class that implements VLM-based detection using CLIP. Responsible for:
- Loading and preprocessing images
- Generating object proposals
- Calculating text-image similarities
- Returning bounding boxes and confidence scores

### MAIC™ Consciousness

Implements the artificial consciousness system that:
- Maintains an internal state of attention and confidence
- Performs self-reflection on detections
- Generates insights based on accumulated experience
- Adjusts internal parameters based on results

### HIM™ Module

Implements the hybrid intelligence module that:
- Maintains a detection history by object type
- Adjusts confidence thresholds based on historical performance
- Activates adaptive learning after sufficient detections

### MAIC™ Memory

Manages persistent storage of:
- Detection history
- Consciousness states
- Generated insights
- Adaptive configurations

## System Messages

The system provides detailed feedback during detection:

- **Detection Results**: Shows prompt, object count, coordinates, and confidence
- **MAIC Consciousness State**: Displays attention focus, confidence level, uncertainty, and reflection depth
- **HIM Adaptive Learning Status**: Shows current confidence threshold and detection history
- **Low Confidence Messages**: Alerts when average confidence is low or no objects are detected

## Limitations and Considerations

- Performance depends on image quality and prompt clarity
- Very small or partially visible objects may be difficult to detect
- The system works best with specific and descriptive prompts
- Initial CLIP model loading may take some time on resource-limited devices

## Installation Requirements

````
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python matplotlib numpy
```

## Project Structure

```
.
├── CyberTechVLMDetector.py     # Main script of the system
├── input/                      # Folder containing input images
│   └── VLM_Scenario-image.jpeg
├── output/                     # Folder where results are saved
├── maic_memory.json            # Persistent memory file
└── README.md                   # This file
```

## Conclusion

The CyberTech VLM Detector demonstrates how vision-language models can be used to build flexible and generalizable object detection systems that run entirely on edge devices, without relying on traditional anchor-based detectors or cloud services.

## Contact

For questions or issues, [https://linkedin.com/in/hellodav](https://linkedin.com/in/hellodav). Please refer to the code documentation and comments within the implementation files for more details.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright David C Cavalcante
