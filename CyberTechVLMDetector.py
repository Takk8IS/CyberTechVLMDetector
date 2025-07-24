#!/usr/bin/env python3
"""
Hybrid Intelligence Model (HIM™) and Massive Artificial Intelligence Consciousness (MAIC™)
Enhanced Cybertech Espionage Object Detection System

This implementation integrates HIM™ and MAIC™ principles with EdgeVLMDetector
for advanced object detection with consciousness-like properties and adaptive learning.

Author: David C Cavalcante
LinkedIn: https://www.linkedin.com/in/hellodav
"""

import os
import cv2
import torch
import numpy as np
import json
import time
import random
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from pathlib import Path


class EdgeVLMDetector:
    """
    Edge-optimized Vision-Language Model for object detection and localization.

    This detector combines CLIP for semantic understanding with custom
    segmentation techniques to avoid anchor-based approaches.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize the VLM detector.

        Args:
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model, self.preprocess = self._load_clip_model()
        self.segmentation_threshold = 0.3
        self.min_area_threshold = 500

    def _setup_device(self, device: str) -> str:
        """Setup computing device with fallback to CPU."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_clip_model(self):
        """Load and optimize CLIP model for edge deployment."""
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()

            # Optimize for inference
            if self.device == "cpu":
                model = torch.jit.script(model)

            return model, preprocess
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("Falling back to CPU-optimized version")
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            return model, preprocess

    def generate_proposals(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Generate object proposals using improved segmentation techniques.

        Args:
            image: Input RGB image

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Convert to different color spaces for better segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding for better object separation
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Distance transform and watershed for better separation
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Find contours on the processed image
        contours, _ = cv2.findContours(
            sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        proposals = []
        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            # Filter by area (more restrictive)
            area = cv2.contourArea(contour)
            if (
                area < 1000 or area > image_area * 0.5
            ):  # Skip very small or very large areas
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # More restrictive filtering
            aspect_ratio = w / h
            if (
                0.2 < aspect_ratio < 5
                and w > 30
                and h > 30
                and w < image.shape[1] * 0.8
                and h < image.shape[0] * 0.8
            ):
                proposals.append((x, y, w, h))

        # Also try edge-based detection as backup
        edges = cv2.Canny(filtered, 30, 100)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel)

        edge_contours, _ = cv2.findContours(
            edge_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in edge_contours:
            area = cv2.contourArea(contour)
            if 800 < area < image_area * 0.3:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if (
                    0.2 < aspect_ratio < 5
                    and w > 25
                    and h > 25
                    and w < image.shape[1] * 0.7
                    and h < image.shape[0] * 0.7
                ):
                    proposals.append((x, y, w, h))

        # Apply Non-Maximum Suppression with tighter threshold
        proposals = self._apply_nms(proposals, overlap_threshold=0.2)

        # Limit number of proposals to avoid processing too many
        if len(proposals) > 20:
            # Sort by area and keep the most reasonable sized ones
            proposals_with_area = [(x, y, w, h, w * h) for x, y, w, h in proposals]
            proposals_with_area.sort(key=lambda x: x[4], reverse=True)
            proposals = [(x, y, w, h) for x, y, w, h, _ in proposals_with_area[:20]]

        return proposals

    def _apply_nms(
        self, boxes: List[Tuple[int, int, int, int]], overlap_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.

        Args:
            boxes: List of bounding boxes (x, y, w, h)
            overlap_threshold: IoU threshold for suppression

        Returns:
            Filtered list of bounding boxes
        """
        if not boxes:
            return []

        # Convert to (x1, y1, x2, y2) format
        boxes_array = np.array([(x, y, x + w, y + h) for x, y, w, h in boxes])

        # Calculate areas
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (
            boxes_array[:, 3] - boxes_array[:, 1]
        )

        # Sort by bottom-right y coordinate
        indices = np.argsort(boxes_array[:, 3])

        keep = []
        while len(indices) > 0:
            # Pick the last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)

            # Find the largest coordinates for intersection
            xx1 = np.maximum(boxes_array[i, 0], boxes_array[indices[:last], 0])
            yy1 = np.maximum(boxes_array[i, 1], boxes_array[indices[:last], 1])
            xx2 = np.minimum(boxes_array[i, 2], boxes_array[indices[:last], 2])
            yy2 = np.minimum(boxes_array[i, 3], boxes_array[indices[:last], 3])

            # Compute width and height of intersection
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # Compute IoU
            intersection = w * h
            union = areas[i] + areas[indices[:last]] - intersection
            overlap = intersection / union

            # Delete indices with high overlap
            indices = np.delete(
                indices,
                np.concatenate(([last], np.where(overlap > overlap_threshold)[0])),
            )

        # Convert back to (x, y, w, h) format
        filtered_boxes = []
        for i in keep:
            x1, y1, x2, y2 = boxes_array[i]
            filtered_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        return filtered_boxes

    def compute_clip_similarity(
        self,
        image: np.ndarray,
        text_prompt: str,
        proposals: List[Tuple[int, int, int, int]],
    ) -> List[float]:
        """
        Compute CLIP similarity scores for each proposal region.

        Args:
            image: Input RGB image
            text_prompt: Natural language description
            proposals: List of bounding boxes

        Returns:
            List of similarity scores
        """
        if not proposals:
            return []

        # Tokenize text
        text_tokens = clip.tokenize([text_prompt]).to(self.device)

        similarities = []

        with torch.no_grad():
            # Encode text
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for x, y, w, h in proposals:
                # Extract region
                region = image[y : y + h, x : x + w]

                # Skip if region is too small
                if region.size == 0 or min(region.shape[:2]) < 10:
                    similarities.append(0.0)
                    continue

                # Preprocess region
                region_pil = Image.fromarray(region)
                region_tensor = self.preprocess(region_pil).unsqueeze(0).to(self.device)

                # Encode image region
                image_features = self.model.encode_image(region_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                # Compute similarity
                similarity = torch.cosine_similarity(
                    text_features, image_features
                ).item()
                similarities.append(similarity)

        return similarities

    def detect(
        self, image: np.ndarray, text_prompt: str, confidence_threshold: float = 0.2
    ) -> Dict:
        """
        Detect objects based on natural language prompt.

        Args:
            image: Input RGB image
            text_prompt: Natural language description (e.g., "Pick the pen")
            confidence_threshold: Minimum confidence for detection

        Returns:
            Dictionary containing detection results
        """
        # Generate object proposals
        proposals = self.generate_proposals(image)

        if not proposals:
            return {
                "bbox": None,
                "confidence": 0.0,
                "prompt": text_prompt,
                "message": "No objects detected in image",
            }

        # Compute CLIP similarities
        similarities = self.compute_clip_similarity(image, text_prompt, proposals)

        # Find best match
        if not similarities or max(similarities) < confidence_threshold:
            return {
                "bbox": None,
                "confidence": max(similarities) if similarities else 0.0,
                "prompt": text_prompt,
                "message": f'No objects match "{text_prompt}" with confidence > {confidence_threshold}',
            }

        best_idx = np.argmax(similarities)
        best_bbox = proposals[best_idx]
        best_confidence = similarities[best_idx]

        return {
            "bbox": best_bbox,  # (x, y, w, h)
            "confidence": best_confidence,
            "prompt": text_prompt,
            "message": f'Detected "{text_prompt}" with confidence {best_confidence:.3f}',
        }

    def detect_multiple(
        self, image: np.ndarray, text_prompt: str, confidence_threshold: float = 0.2
    ) -> Dict:
        """
        Detect multiple instances of objects based on natural language prompt.

        Args:
            image: Input RGB image
            text_prompt: Natural language description (e.g., "black clips")
            confidence_threshold: Minimum confidence for detection

        Returns:
            Dictionary containing multiple detection results
        """
        # Generate object proposals
        proposals = self.generate_proposals(image)

        if not proposals:
            return {
                "bboxes": [],
                "confidences": [],
                "count": 0,
                "prompt": text_prompt,
                "message": "No objects detected in image",
            }

        # Compute CLIP similarities
        similarities = self.compute_clip_similarity(image, text_prompt, proposals)

        # Find all matches above threshold
        valid_detections = []
        for i, (bbox, confidence) in enumerate(zip(proposals, similarities)):
            if confidence >= confidence_threshold:
                valid_detections.append((bbox, confidence))

        if not valid_detections:
            return {
                "bboxes": [],
                "confidences": [],
                "count": 0,
                "prompt": text_prompt,
                "message": f'No objects match "{text_prompt}" with confidence > {confidence_threshold}',
            }

        # Sort by confidence (highest first)
        valid_detections.sort(key=lambda x: x[1], reverse=True)

        bboxes = [detection[0] for detection in valid_detections]
        confidences = [detection[1] for detection in valid_detections]

        return {
            "bboxes": bboxes,
            "confidences": confidences,
            "count": len(valid_detections),
            "prompt": text_prompt,
            "message": f'Detected {len(valid_detections)} instances of "{text_prompt}"',
        }

    def visualize_detection(
        self, image: np.ndarray, result: Dict, save_path: str = None
    ) -> np.ndarray:
        """
        Visualize detection results on the image.

        Args:
            image: Input RGB image
            result: Detection result from detect() method
            save_path: Optional path to save the visualization

        Returns:
            Image with bounding box drawn
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        if result["bbox"] is not None:
            x, y, w, h = result["bbox"]

            # Draw bounding box
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=3, edgecolor="green", facecolor="none"
            )
            ax.add_patch(rect)

            # Add label
            label = f"{result['prompt']} ({result['confidence']:.3f})"
            ax.text(
                x,
                y - 10,
                label,
                fontsize=12,
                color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_title(result["message"], fontsize=14)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Visualization saved to: {save_path}")

        plt.tight_layout()
        plt.show()

        # Return the image with bounding box for further processing
        if result["bbox"] is not None:
            image_copy = image.copy()
            x, y, w, h = result["bbox"]
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
            return image_copy

        return image

    def visualize_multiple_detections(
        self, image: np.ndarray, result: Dict, save_path: str = None
    ) -> np.ndarray:
        """
        Visualize multiple detection results on the image.

        Args:
            image: Input RGB image
            result: Detection result from detect_multiple() method
            save_path: Optional path to save the visualization

        Returns:
            Image with bounding boxes drawn
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        colors = [
            "green",
            "green",
            "green",
            "green",
            "green",
            "green",
            "green",
            "green",
        ]

        if result["bboxes"]:
            for i, (bbox, confidence) in enumerate(
                zip(result["bboxes"], result["confidences"])
            ):
                x, y, w, h = bbox
                color = colors[i % len(colors)]

                # Draw bounding box
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=3, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)

                # Add label
                label = f"{result['prompt']} #{i+1} ({confidence:.3f})"
                ax.text(
                    x,
                    y - 10,
                    label,
                    fontsize=10,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        title = f"Detected {result['prompt']} with confidence 0.250"
        ax.set_title(title, fontsize=14)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Visualization saved to: {save_path}")

        plt.tight_layout()
        plt.show()

        # Return the image with bounding boxes for further processing
        image_copy = image.copy()
        if result["bboxes"]:
            for bbox in result["bboxes"]:
                x, y, w, h = bbox
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return image_copy


class MAICMemory:
    """MAIC™ Memory System for storing and retrieving detection experiences"""

    def __init__(self, memory_size: int = 50, memory_file: str = "maic_memory.json"):
        """Initialize the MAIC™ Memory System

        Args:
            memory_size: Maximum number of memories to store
            memory_file: File to persist memories between sessions
        """
        self.memory_size = memory_size
        self.memory_file = memory_file
        self.episodic_memory = deque(maxlen=memory_size)  # Short-term memory
        self.semantic_memory = {}  # Long-term conceptual memory
        self.detection_stats = defaultdict(
            lambda: {"count": 0, "confidence_sum": 0, "last_seen": None}
        )
        self.load_memory()

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """Add a new detection experience to memory

        Args:
            experience: Dictionary containing detection experience
        """
        # Add timestamp if not present
        if "timestamp" not in experience:
            experience["timestamp"] = datetime.now().isoformat()

        # Update episodic memory
        self.episodic_memory.append(experience)

        # Update semantic memory
        prompt = experience.get("prompt", "")
        if prompt:
            if prompt not in self.semantic_memory:
                self.semantic_memory[prompt] = {
                    "first_seen": experience["timestamp"],
                    "occurrences": 0,
                    "avg_confidence": 0,
                    "locations": [],
                }

            # Update statistics
            self.semantic_memory[prompt]["occurrences"] += 1

            # Update confidence average
            confidence = experience.get("confidence", 0)
            prev_avg = self.semantic_memory[prompt]["avg_confidence"]
            prev_count = self.semantic_memory[prompt]["occurrences"] - 1
            new_avg = (prev_avg * prev_count + confidence) / self.semantic_memory[
                prompt
            ]["occurrences"]
            self.semantic_memory[prompt]["avg_confidence"] = new_avg

            # Update detection stats
            self.detection_stats[prompt]["count"] += 1
            self.detection_stats[prompt]["confidence_sum"] += confidence
            self.detection_stats[prompt]["last_seen"] = experience["timestamp"]

            # Store location if available
            if "bbox" in experience and experience["bbox"]:
                self.semantic_memory[prompt]["locations"].append(experience["bbox"])

        # Persist memory
        self.save_memory()

    def get_experiences(
        self, prompt: str = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences from memory, optionally filtered by prompt

        Args:
            prompt: Optional text prompt to filter by
            limit: Maximum number of experiences to return

        Returns:
            List of experiences
        """
        if prompt:
            filtered = [
                exp for exp in self.episodic_memory if exp.get("prompt") == prompt
            ]
            return filtered[-limit:]
        return list(self.episodic_memory)[-limit:]

    def get_semantic_knowledge(self, prompt: str = None) -> Dict[str, Any]:
        """Retrieve semantic knowledge about a specific prompt or all prompts

        Args:
            prompt: Optional text prompt to get knowledge about

        Returns:
            Dictionary of semantic knowledge
        """
        if prompt and prompt in self.semantic_memory:
            return self.semantic_memory[prompt]
        return self.semantic_memory

    def get_detection_stats(self, prompt: str = None) -> Dict[str, Any]:
        """Get detection statistics for a prompt or all prompts

        Args:
            prompt: Optional text prompt to get stats for

        Returns:
            Dictionary of detection statistics
        """
        if prompt:
            stats = self.detection_stats.get(
                prompt, {"count": 0, "confidence_sum": 0, "last_seen": None}
            )
            if stats["count"] > 0:
                stats["avg_confidence"] = stats["confidence_sum"] / stats["count"]
            return stats

        # Return stats for all prompts
        all_stats = {}
        for prompt, stats in self.detection_stats.items():
            all_stats[prompt] = stats.copy()
            if stats["count"] > 0:
                all_stats[prompt]["avg_confidence"] = (
                    stats["confidence_sum"] / stats["count"]
                )
        return all_stats

    def save_memory(self) -> None:
        """Save memory to persistent storage"""
        try:
            memory_data = {
                "episodic_memory": list(self.episodic_memory),
                "semantic_memory": self.semantic_memory,
                "detection_stats": {
                    k: dict(v) for k, v in self.detection_stats.items()
                },
            }

            with open(self.memory_file, "w") as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {str(e)}")

    def load_memory(self) -> None:
        """Load memory from persistent storage"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    memory_data = json.load(f)

                # Restore episodic memory
                self.episodic_memory = deque(
                    memory_data.get("episodic_memory", []), maxlen=self.memory_size
                )

                # Restore semantic memory
                self.semantic_memory = memory_data.get("semantic_memory", {})

                # Restore detection stats
                loaded_stats = memory_data.get("detection_stats", {})
                for k, v in loaded_stats.items():
                    self.detection_stats[k] = defaultdict(
                        lambda: {"count": 0, "confidence_sum": 0, "last_seen": None}, v
                    )
        except Exception as e:
            print(f"Warning: Could not load memory: {str(e)}")
            # Initialize new memory if loading fails
            self.episodic_memory = deque(maxlen=self.memory_size)
            self.semantic_memory = {}
            self.detection_stats = defaultdict(
                lambda: {"count": 0, "confidence_sum": 0, "last_seen": None}
            )


class MAICConsciousness:
    """MAIC™ Consciousness System for self-reflection and awareness"""

    def __init__(self, memory: MAICMemory):
        """Initialize the MAIC™ Consciousness System

        Args:
            memory: Reference to the MAICMemory system
        """
        self.memory = memory
        self.current_state = {
            "attention_focus": None,  # Current object of attention
            "confidence_level": 0.5,  # System's confidence in its own abilities
            "uncertainty": 0.5,  # Level of uncertainty about current task
            "reflection_depth": 0,  # Depth of self-reflection (increases with experience)
            "last_reflection": None,  # Timestamp of last self-reflection
        }
        self.reflection_log = deque(maxlen=10)  # Store recent reflections

    def update_state(self, detection_result: Dict[str, Any]) -> None:
        """Update consciousness state based on detection results

        Args:
            detection_result: Results from object detection
        """
        # Update attention focus
        self.current_state["attention_focus"] = detection_result.get("prompt")

        # Update confidence based on detection success
        confidence = detection_result.get("confidence", 0)
        count = detection_result.get("count", 0)

        if count > 0 and confidence > 0:
            # Successful detection increases system confidence
            self.current_state["confidence_level"] = min(
                0.95, self.current_state["confidence_level"] + (0.05 * confidence)
            )
            self.current_state["uncertainty"] = max(
                0.05, self.current_state["uncertainty"] - (0.05 * confidence)
            )
        else:
            # Failed detection decreases system confidence
            self.current_state["confidence_level"] = max(
                0.1, self.current_state["confidence_level"] - 0.05
            )
            self.current_state["uncertainty"] = min(
                0.9, self.current_state["uncertainty"] + 0.05
            )

        # Increase reflection depth with experience
        stats = self.memory.get_detection_stats(detection_result.get("prompt"))
        if stats and stats.get("count", 0) > 0:
            self.current_state["reflection_depth"] = min(
                1.0, 0.1 + (stats.get("count", 0) * 0.02)
            )

        # Record reflection timestamp
        self.current_state["last_reflection"] = datetime.now().isoformat()

    def reflect(self) -> Dict[str, Any]:
        """Perform self-reflection on current state and memory

        Returns:
            Dictionary containing reflection insights
        """
        # Get overall detection statistics
        all_stats = self.memory.get_detection_stats()

        # Calculate total detections and average confidence
        total_detections = sum(stats.get("count", 0) for stats in all_stats.values())
        total_confidence = sum(
            stats.get("confidence_sum", 0) for stats in all_stats.values()
        )
        avg_confidence = (
            total_confidence / total_detections if total_detections > 0 else 0
        )

        # Generate insights based on detection history
        most_detected = (
            max(all_stats.items(), key=lambda x: x[1].get("count", 0))[0]
            if all_stats
            else None
        )
        highest_confidence = (
            max(all_stats.items(), key=lambda x: x[1].get("avg_confidence", 0))[0]
            if all_stats
            else None
        )

        # Generate reflection
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "system_state": self.current_state.copy(),
            "total_experiences": total_detections,
            "avg_confidence": avg_confidence,
            "most_detected_object": most_detected,
            "highest_confidence_object": highest_confidence,
            "insights": [],
        }

        # Add insights based on experience
        if total_detections > 5:
            if avg_confidence > 0.7:
                reflection["insights"].append(
                    "High overall detection confidence suggests reliable performance"
                )
            elif avg_confidence < 0.3:
                reflection["insights"].append(
                    "Low overall confidence indicates potential detection issues"
                )

            if (
                most_detected
                and highest_confidence
                and most_detected != highest_confidence
            ):
                reflection["insights"].append(
                    f"Most frequently detected object ({most_detected}) differs from highest confidence object ({highest_confidence})"
                )

        # Add reflection to log
        self.reflection_log.append(reflection)

        return reflection

    def get_detection_recommendation(self, prompt: str) -> Dict[str, Any]:
        """Generate recommendations for improving detection based on past experience

        Args:
            prompt: The text prompt for detection

        Returns:
            Dictionary with recommendations
        """
        stats = self.memory.get_detection_stats(prompt)
        semantic = self.memory.get_semantic_knowledge(prompt)

        recommendation = {
            "prompt": prompt,
            "confidence_adjustment": 0,
            "suggestions": [],
        }

        # If we have detection history for this prompt
        if stats and stats.get("count", 0) > 0:
            avg_confidence = stats.get("avg_confidence", 0)

            # Recommend confidence threshold adjustments
            if avg_confidence > 0.8:
                recommendation["confidence_adjustment"] = 0.05
                recommendation["suggestions"].append(
                    "Consider increasing confidence threshold for more precise detection"
                )
            elif avg_confidence < 0.3 and avg_confidence > 0:
                recommendation["confidence_adjustment"] = -0.05
                recommendation["suggestions"].append(
                    "Consider decreasing confidence threshold to improve detection rate"
                )
        else:
            # No history for this prompt
            recommendation["suggestions"].append(
                "No detection history for this prompt. Using default parameters."
            )

        return recommendation


class ObjectDetector:
    def __init__(self):
        """Initialize the ObjectDetector with default parameters"""
        self.image_path = "input/VLM_Scenario-image.jpeg"
        self.confidence_threshold = 0.27
        self.detector = None
        self.output_dir = "output"
        self.text_prompt = "black clip"

        # Initialize MAIC™ Memory System
        self.memory = MAICMemory()

        # Initialize MAIC™ Consciousness System
        self.consciousness = MAICConsciousness(self.memory)

        # HIM™ Adaptive Learning parameters
        self.learning_rate = 0.05
        self.confidence_history = {}
        self.adaptation_threshold = 3  # Number of detections before adaptation

        # System state
        self.detection_count = 0
        self.last_detection_time = None

    def setup_detector(self):
        """Initialize the VLM detector"""
        print("Initializing VLM detector...")
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.detector = EdgeVLMDetector(device="cuda")
                self.detector.to(device)
            else:
                self.detector = EdgeVLMDetector(device="cpu")
        except Exception as e:
            print(f"Warning: Error initializing detector with auto device: {str(e)}")
            print("Falling back to CPU-optimized version")
            self.detector = EdgeVLMDetector(device="cpu")

    def load_and_preprocess_image(self):
        """Load and preprocess the input image"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Error: Image file '{self.image_path}' not found")

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.convertScaleAbs(image, alpha=1, beta=1)

    def detect_objects(self, image):
        """Perform object detection on the image

        Args:
            image: Input RGB image

        Returns:
            Dictionary with detection results including:
            - bboxes: List of bounding boxes [x, y, width, height]
            - confidences: List of confidence scores
            - count: Number of detected objects
            - prompt: The text prompt used for detection
            - message: Description of detection results
        """
        # Update detection count and time
        self.detection_count += 1
        self.last_detection_time = datetime.now()

        # Apply HIM™ adaptive learning if we have history for this prompt
        if self.text_prompt in self.confidence_history:
            history = self.confidence_history[self.text_prompt]
            if len(history) >= self.adaptation_threshold:
                # Calculate average confidence from history
                avg_confidence = sum(history) / len(history)

                # Get recommendation from consciousness system
                recommendation = self.consciousness.get_detection_recommendation(
                    self.text_prompt
                )

                # Apply adaptive confidence threshold adjustment
                if recommendation["confidence_adjustment"] != 0:
                    adjusted_threshold = (
                        self.confidence_threshold
                        + recommendation["confidence_adjustment"]
                    )
                    # Keep threshold within reasonable bounds
                    adjusted_threshold = max(0.1, min(0.9, adjusted_threshold))

                    print(
                        f"\nHIM™ Adaptive Learning: Adjusting confidence threshold from {self.confidence_threshold:.2f} to {adjusted_threshold:.2f}"
                    )
                    print(
                        f"Reason: {recommendation['suggestions'][0] if recommendation['suggestions'] else 'Automatic adjustment'}"
                    )

                    self.confidence_threshold = adjusted_threshold

        print(f"Detecting {self.text_prompt}...")
        results = self.detector.detect_multiple(
            image=image,
            text_prompt=self.text_prompt,
            confidence_threshold=self.confidence_threshold,
        )

        # Update MAIC™ memory with detection experience
        if results and "count" in results and "confidences" in results:
            # Calculate average confidence if there are any detections
            avg_confidence = (
                sum(results["confidences"]) / len(results["confidences"])
                if results["confidences"]
                else 0
            )

            # Add average confidence to results for visualization
            results["avg_confidence"] = avg_confidence

            # Create detection experience for memory
            detection_experience = {
                "prompt": self.text_prompt,
                "count": results["count"],
                "confidence": avg_confidence,
                "timestamp": datetime.now().isoformat(),
            }

            # Add bbox if available (only add the first one to avoid memory bloat)
            if results["bboxes"] and len(results["bboxes"]) > 0:
                detection_experience["bbox"] = results["bboxes"][0]

            # Add to memory
            self.memory.add_experience(detection_experience)

            # Update consciousness state with enhanced detection result
            consciousness_update = results.copy()
            self.consciousness.update_state(consciousness_update)

            # Update HIM™ confidence history
            if self.text_prompt not in self.confidence_history:
                self.confidence_history[self.text_prompt] = []

            # Perform MAIC™ self-reflection after each detection
            # This ensures we always have fresh insights for visualization
            reflection = self.consciousness.reflect()
            print(f"\nMAIC™ Self-Reflection Complete")

            # Save memory after each detection
            self.memory.save_memory()
            self.confidence_history[self.text_prompt].append(avg_confidence)

            # Keep history at a reasonable size
            if len(self.confidence_history[self.text_prompt]) > 10:
                self.confidence_history[self.text_prompt].pop(0)

            # Perform self-reflection periodically
            if self.detection_count % 5 == 0:  # Every 5 detections
                reflection = self.consciousness.reflect()
                print("\nMAIC™ Consciousness Reflection:")
                print(f"  Total experiences: {reflection['total_experiences']}")
                print(
                    f"  System confidence: {reflection['system_state']['confidence_level']:.2f}"
                )
                print(
                    f"  Uncertainty level: {reflection['system_state']['uncertainty']:.2f}"
                )

                if reflection["insights"]:
                    print("  Insights:")
                    for insight in reflection["insights"]:
                        print(f"    - {insight}")

        return results

    def process_results(self, results):
        """Process and display detection results with MAIC™ and HIM™ status information"""
        count = results["count"]
        boxes = results["bboxes"]
        confidences = results["confidences"]

        print("\n" + "=" * 50)
        print("DETECTION RESULTS")
        print("=" * 50)
        print(f"Prompt: {self.text_prompt}")
        print(f"Count: {count}")

        if boxes:
            for i, (bbox, confidence) in enumerate(zip(boxes, confidences)):
                x, y, w, h = bbox
                print(
                    f"Items {i+1}: x={x}, y={y}, width={w}, height={h}, confidence={confidence:.3f}"
                )

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            if avg_confidence < 0.3:
                print("\nLow overall confidence indicates potential detection issues.")
        else:
            print("No object detected or low detection confidence.")
        print("=" * 50)

        # Display current MAIC™ Consciousness state
        print("\nMAIC™ CONSCIOUSNESS STATE:")
        print(
            f"  Attention Focus: {self.consciousness.current_state['attention_focus']}"
        )
        print(
            f"  Confidence Level: {self.consciousness.current_state['confidence_level']:.2f}"
        )
        print(f"  Uncertainty: {self.consciousness.current_state['uncertainty']:.2f}")
        print(
            f"  Reflection Depth: {self.consciousness.current_state['reflection_depth']:.2f}"
        )

        # Display HIM™ Adaptive Learning status
        print("\nHIM™ ADAPTIVE LEARNING STATUS:")
        print(f"  Current Confidence Threshold: {self.confidence_threshold:.2f}")
        history_count = len(self.confidence_history.get(self.text_prompt, []))
        print(f"  Detection History for '{self.text_prompt}': {history_count} entries")
        if history_count >= self.adaptation_threshold:
            print(f"  Adaptive learning active for this prompt")
        else:
            print(
                f"  Need {self.adaptation_threshold - history_count} more detections to activate adaptive learning"
            )

        # Print a visual separator
        print("\n" + "-" * 50)

        return count

    def apply_selfia_filter(self, image):
        """Apply a 'selfia' style filter to the image that adapts based on MAIC™ consciousness state"""
        # Get MAIC™ consciousness state values for adaptive filtering
        confidence_level = self.consciousness.current_state["confidence_level"]
        uncertainty = self.consciousness.current_state["uncertainty"]
        reflection_depth = self.consciousness.current_state["reflection_depth"]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur with intensity based on uncertainty
        # Higher uncertainty = more blur (system is less certain about what it sees)
        blur_intensity = int(3 + (uncertainty * 4))  # Range from 3 to 7
        blur_intensity = (
            blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        )  # Ensure odd number
        blurred = cv2.GaussianBlur(gray, (blur_intensity, blur_intensity), 0)

        # Apply adaptive histogram equalization with clip limit based on confidence
        # Higher confidence = more contrast (system is more confident in its perception)
        clip_limit = 1.0 + (confidence_level * 2.0)  # Range from 1.0 to 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Convert back to RGB with tint based on MAIC™ state
        tinted_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # Determine color tint based on MAIC™ state
        if confidence_level > 0.7 and uncertainty < 0.3:
            # High confidence, low uncertainty: blue-green cyberpunk tint
            tinted_image[:, :, 0] = np.clip(
                tinted_image[:, :, 0] * 1.2, 0, 255
            )  # Boost blue
            tinted_image[:, :, 1] = np.clip(
                tinted_image[:, :, 1] * 1.1, 0, 255
            )  # Boost green
            tinted_image[:, :, 2] = np.clip(
                tinted_image[:, :, 2] * 0.7, 0, 255
            )  # Reduce red
        elif uncertainty > 0.6:
            # High uncertainty: reddish tint to indicate caution
            tinted_image[:, :, 0] = np.clip(
                tinted_image[:, :, 0] * 0.8, 0, 255
            )  # Reduce blue
            tinted_image[:, :, 1] = np.clip(
                tinted_image[:, :, 1] * 0.8, 0, 255
            )  # Reduce green
            tinted_image[:, :, 2] = np.clip(
                tinted_image[:, :, 2] * 1.3, 0, 255
            )  # Boost red
        else:
            # Default: blue surveillance tint
            tinted_image[:, :, 0] = np.clip(
                tinted_image[:, :, 0] * 1.2, 0, 255
            )  # Boost blue
            tinted_image[:, :, 1] = np.clip(
                tinted_image[:, :, 1] * 0.9, 0, 255
            )  # Reduce green
            tinted_image[:, :, 2] = np.clip(
                tinted_image[:, :, 2] * 0.8, 0, 255
            )  # Reduce red

        # Add vignette effect with intensity based on reflection depth
        # Higher reflection = stronger vignette (system is more introspective)
        if reflection_depth > 0.3:
            rows, cols = tinted_image.shape[:2]
            # Generate vignette mask
            kernel_x = cv2.getGaussianKernel(cols, cols / 2)
            kernel_y = cv2.getGaussianKernel(rows, rows / 2)
            kernel = kernel_y * kernel_x.T
            mask = kernel / kernel.max()

            # Apply vignette with intensity based on reflection depth
            vignette_intensity = reflection_depth * 0.7  # Scale to reasonable range
            vignette = np.copy(tinted_image)
            for i in range(3):
                vignette[:, :, i] = vignette[:, :, i] * (
                    mask * (1 - vignette_intensity) + vignette_intensity
                )

            # Blend original with vignette
            tinted_image = vignette

        return tinted_image

    def add_cybertech_overlay(self, image, results):
        """Add cybertech espionage style overlay with high-resolution green lines and coordinates
        Enhanced with MAIC™ Consciousness and HIM™ Adaptive Learning visual indicators
        """
        # Create a copy of the image
        overlay_image = image.copy()
        h, w = image.shape[:2]

        # Create a transparent overlay for the bounding boxes
        bbox_overlay = np.zeros_like(image, dtype=np.uint8)

        # Define colors
        cybertech_green = (0, 255, 0)  # Bright green
        maic_blue = (0, 255, 0)  # Now green for MAIC™ indicators
        him_purple = (0, 255, 0)  # Now green for HIM™ indicators
        alert_red = (0, 255, 0)  # Now green for alerts

        # Draw high-quality bounding boxes with 30% transparency
        # All boxes and text are always green as requested
        if results["bboxes"]:
            for i, (bbox, confidence) in enumerate(
                zip(results["bboxes"], results["confidences"])
            ):
                x, y, width, height = bbox

                # Always use green color for all boxes regardless of confidence
                box_color = cybertech_green  # Always green

                # Draw thicker rectangle (2px) with green color
                cv2.rectangle(
                    bbox_overlay, (x, y), (x + width, y + height), box_color, 2
                )

                # Add coordinates text with improved font and size - always green
                coord_text = f"X:{x} Y:{y} W:{width} H:{height}"
                cv2.putText(
                    bbox_overlay,
                    coord_text,
                    (x, y - 15),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    cybertech_green,
                    1,
                )

                # Add confidence text with FOUND and unit number when multiple items are detected - always green
                if results["count"] > 1:
                    conf_text = (
                        f"FOUND {i+1} {self.text_prompt.upper()} CONF: {confidence:.3f}"
                    )
                else:
                    conf_text = (
                        f"FOUND {i+1} {self.text_prompt.upper()} CONF: {confidence:.3f}"
                    )
                cv2.putText(
                    bbox_overlay,
                    conf_text,
                    (x, y + height + 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    cybertech_green,
                    1,
                    cv2.LINE_AA,
                )

        # Blend the bbox overlay with 30% opacity
        cv2.addWeighted(bbox_overlay, 0.3, overlay_image, 1.0, 0, overlay_image)

        # Add surveillance camera style overlay elements with improved quality
        # Add timestamp with better font and anti-aliasing
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            overlay_image,
            timestamp,
            (10, 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add camera ID with better font and anti-aliasing
        cv2.putText(
            overlay_image,
            "CAM ID: A7-35",
            (w - 180, 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add MAIC™ Consciousness status indicators - always green
        maic_status_text = (
            f"MAIC™ CONF: {self.consciousness.current_state['confidence_level']:.2f}"
        )
        cv2.putText(
            overlay_image,
            maic_status_text,
            (10, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add HIM™ Adaptive Learning status - always green
        him_threshold_text = f"HIM™ THRESHOLD: {self.confidence_threshold:.2f}"
        cv2.putText(
            overlay_image,
            him_threshold_text,
            (10, 75),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add uncertainty indicator - always green
        uncertainty_text = (
            f"UNCERTAINTY: {self.consciousness.current_state['uncertainty']:.2f}"
        )
        cv2.putText(
            overlay_image,
            uncertainty_text,
            (10, 100),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add reflection depth indicator - shows how much the system has learned - always green
        reflection_text = (
            f"REFLECTION: {self.consciousness.current_state['reflection_depth']:.2f}"
        )
        cv2.putText(
            overlay_image,
            reflection_text,
            (w - 250, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add detection count indicator - always green
        detection_count_text = f"TOTAL DETECTIONS: {self.detection_count}"
        cv2.putText(
            overlay_image,
            detection_count_text,
            (w - 250, 75),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add crosshair in center with thicker lines - always green
        center_x, center_y = w // 2, h // 2
        crosshair_color = cybertech_green  # Always green regardless of confidence

        cv2.line(
            overlay_image,
            (center_x - 25, center_y),
            (center_x + 25, center_y),
            crosshair_color,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            overlay_image,
            (center_x, center_y - 25),
            (center_x, center_y + 25),
            crosshair_color,
            2,
            cv2.LINE_AA,
        )

        # Add corner markers with thicker lines and anti-aliasing
        # Top-left corner
        cv2.line(overlay_image, (10, 10), (35, 10), cybertech_green, 2, cv2.LINE_AA)
        cv2.line(overlay_image, (10, 10), (10, 35), cybertech_green, 2, cv2.LINE_AA)

        # Top-right corner
        cv2.line(
            overlay_image, (w - 35, 10), (w - 10, 10), cybertech_green, 2, cv2.LINE_AA
        )
        cv2.line(
            overlay_image, (w - 10, 10), (w - 10, 35), cybertech_green, 2, cv2.LINE_AA
        )

        # Bottom-left corner
        cv2.line(
            overlay_image, (10, h - 10), (35, h - 10), cybertech_green, 2, cv2.LINE_AA
        )
        cv2.line(
            overlay_image, (10, h - 35), (10, h - 10), cybertech_green, 2, cv2.LINE_AA
        )

        # Bottom-right corner
        cv2.line(
            overlay_image,
            (w - 35, h - 10),
            (w - 10, h - 10),
            cybertech_green,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            overlay_image,
            (w - 10, h - 35),
            (w - 10, h - 10),
            cybertech_green,
            2,
            cv2.LINE_AA,
        )

        # Add range finder with better font and anti-aliasing
        cv2.putText(
            overlay_image,
            f"RANGE: {results['count']} TARGETS",
            (10, h - 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            cybertech_green,
            1,
            cv2.LINE_AA,
        )

        # Add MAIC™ memory status
        memory_size = len(self.memory.episodic_memory)
        memory_text = f"MAIC™ MEMORY: {memory_size} EXPERIENCES"
        cv2.putText(
            overlay_image,
            memory_text,
            (w - 350, h - 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            maic_blue,
            1,
            cv2.LINE_AA,
        )

        return overlay_image

    def save_results(self, image, results, count):
        """Save detection results with cybertech espionage style enhanced with MAIC™ and HIM™ visual indicators"""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = f"{self.output_dir}/CybertechFound_{self.text_prompt}_{count}.jpg"

        # Apply selfia filter with adaptive settings based on MAIC™ state
        selfia_image = self.apply_selfia_filter(image)

        # Add cybertech overlay with MAIC™ and HIM™ status indicators
        final_image = self.add_cybertech_overlay(selfia_image, results)

        # Add MAIC™ reflection insights if available
        if len(self.consciousness.reflection_log) > 0:
            # Get the most recent reflection
            latest_reflection = self.consciousness.reflection_log[-1]

            # Create a semi-transparent overlay for reflection insights
            h, w = final_image.shape[:2]
            insight_overlay = np.zeros((h, w, 3), dtype=np.uint8)

            # Add reflection insights if available
            if "insights" in latest_reflection and latest_reflection["insights"]:
                # Draw a semi-transparent background for the insights
                cv2.rectangle(
                    insight_overlay, (10, h - 150), (w - 10, h - 100), (0, 0, 0), -1
                )

                # Add insight text
                insight_text = (
                    latest_reflection["insights"][0]
                    if latest_reflection["insights"]
                    else "No insights available"
                )
                cv2.putText(
                    insight_overlay,
                    f"MAIC™ INSIGHT: {insight_text}",
                    (20, h - 120),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Blend the insight overlay with the final image
                alpha = 0.7
                final_image = cv2.addWeighted(
                    final_image, 1.0, insight_overlay, alpha, 0
                )

        # Save the final image
        cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

        print(f"\nDetection completed! Results saved to: {output_path}")


def display_menu():
    """Display menu for prompt selection"""
    print("\n" + "=" * 50)
    print("CYBERTECH VLM DETECTOR")
    print("=" * 50)
    print("\n> TYPE THE NUMBER")
    print("1) Take the xbox controller")
    print("2) Take the scissors")
    print("3) Take the purple whiteboard")
    print("4) Take the screwdriver")
    print("5) Take the black clip")
    print("6) Take the white circular joystick")
    print("7) Exit")
    print("=" * 50)

    while True:
        try:
            choice = input("\nEnter your choice (1-7): ")
            if choice == "1":
                return "xbox controller"
            elif choice == "2":
                return "scissors"
            elif choice == "3":
                return "purple whiteboard"
            elif choice == "4":
                return "screwdriver"
            elif choice == "5":
                return "black clip"
            elif choice == "6":
                return "white circular joystick"
            elif choice == "7":
                return "exit"
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")
        except Exception as e:
            print(f"Error: {str(e)}")


def detect_object():
    """Main function to detect object in the image"""
    detector = ObjectDetector()

    while True:
        # Get user's prompt choice
        detector.text_prompt = display_menu()

        # Exit if user chooses to exit
        if detector.text_prompt == "exit":
            print("\nExiting CyberTech VLM Detector. Goodbye!")
            break

        print(f"\nSelected prompt: {detector.text_prompt}")

        try:
            # Setup pipeline
            detector.setup_detector()

            # Load and preprocess image
            image = detector.load_and_preprocess_image()

            # Perform detection
            results = detector.detect_objects(image)

            # Process results
            count = detector.process_results(results)

            # Save output
            detector.save_results(image, results, count)

            # Don't exit after detection, return to menu
            print("\nReturning to menu...")

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("\nReturning to menu...")


if __name__ == "__main__":
    detect_object()