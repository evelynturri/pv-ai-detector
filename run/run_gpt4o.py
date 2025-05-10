#!/usr/bin/env python3
"""
PV Module Failure Classification using Vision AI

This script analyzes infrared images of photovoltaic modules to classify various types
of failures using OpenAI's vision models. It supports multiple classification modes:
binary (failure/no-failure), detailed multi-class, and reduced classification.

"""

import os
import base64
import json
import random
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configuration - moved to environment variables and CLI arguments
DEFAULT_RANDOM_SEED = 5678
DEFAULT_IMAGE_FOLDER = "dataset/InfraredSolarModules/images"
DEFAULT_METADATA_FILE = 'config/module_metadata.json'
DEFAULT_MAX_TEST_IMAGES = 50  # Limit number of test images to process
DEFAULT_OUTPUT_DIR = "results"

# Import learning maps (assuming this file exists)
try:
    from config.dataset_learning_map import multi_learning_map, multi_reduction_learning_map
except ImportError:
    print("Warning: dataset_learning_map.py not found. Classification mappings may not work correctly.")
    # Define empty maps to allow the script to run
    multi_learning_map = {}
    multi_reduction_learning_map = {}

class PVModuleClassifier:
    """Classifier for PV Module failure detection using vision AI."""
    
    def __init__(self, api_key=None, random_seed=DEFAULT_RANDOM_SEED):
        """Initialize the classifier with API key and settings."""
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set random seed for reproducibility
        self.random_seed = random_seed
        random.seed(self.random_seed)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Create output directory if it doesn't exist
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    @staticmethod
    def encode_image(image_path):
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def get_image_files(folder_path, test_labels):
        """Get filtered list of image files that exist in test_labels."""
        image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]
        test_labels_normalized = set([os.path.splitext(label)[0] for label in test_labels])
        return [f for f in image_files if os.path.splitext(f)[0] in test_labels_normalized]
    
    @staticmethod
    def load_yaml_dataset(file_path):
        """Load a YAML dataset file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Dataset file {file_path} not found.")
            return None
    
    @staticmethod
    def load_metadata(metadata_file=DEFAULT_METADATA_FILE):
        """Load image metadata from JSON file."""
        try:
            with open(metadata_file, 'r') as file:
                data = json.load(file)
            df_json = pd.DataFrame.from_dict(data, orient='index')
            df_json['image_name'] = df_json['image_filepath'].apply(lambda x: x.split('/')[-1])
            return df_json
        except FileNotFoundError:
            print(f"Error: Metadata file {metadata_file} not found.")
            return None
    
    def get_binary_classification(self, encoded_image):
        """Classify image as 'Failure' or 'No_Failure'."""
        messages = [
            {
                "role": "system",
                "content": "You are an image classification system. Your task is to classify infrared images of PV modules as 'Failure' or 'No_Failure', based on if the photovoltaic module has a failure or does not have any failures. Classify them in the following format {'type': 'Failure/No_Failure'}"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this infrared image of a PV module."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                seed=self.random_seed
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def get_detailed_classification(self, encoded_image):
        """Classify image with detailed failure type."""
        messages = [
            {
                "role": "system",
                "content": """You are an image classification system. Your task is to classify infrared images of PV modules as 'Cell', 'Cell-Multi', 'Hot-Spot', 'Hot-Spot-Multi', 'Diode', 'Diode-Multi', 'Offline-Module', based on what type of failure they have. 
                Here is a brief description of the failure patterns: 
                'Cell' is a PV module with single clearly defined rectangular cell appearing hotter than the surrounding cells.
                'Cell-Multi' is a PV module with several distinct cells randomly distributed over the module are noticeably warmer than others.
                'Hot-Spot' is a PV module with a single small point-like, localized, intense heating on one part of the module with the surrounding region gradually transitioning back to a normal temperature.
                'Hot-Spot-Multi' is a PV module with multiple hotspots spread across different areas of the module. Each hotspot appears much warmer than the surrounding areas.
                'Diode' is a PV module with a single bypass diode failure which has the pattern of a single row or column appearing warmer than the rest of the module.
                'Diode-Multi' is a PV module with several large areas or strings of cells appear warm. These inactive regions may span several adjacent rows or sections of the module.
                'Offline-Module' is a PV module with a uniform cool pattern across the entire module with no noticeable hotspots or irregularities.
                The images are colored and blue means cold, red means hot.
                Classify them in the following format {'type': 'Cell/Cell-Multi/Hot-Spot/Hot-Spot-Multi/Diode/Diode-Multi/Offline-Module'}"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this infrared image of a PV module."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                seed=self.random_seed
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def get_reduced_classification(self, encoded_image):
        """Classify image with reduced set of failure types."""
        messages = [
            {
                "role": "system",
                "content": """You are an image classification system. Your task is to classify infrared images of PV modules as 'Offline-Module', 'Cell', 'Cell-Multi', 'Hot-Spot', 'Hot-Spot-Multi', 'Diode', or 'Diode-Multi' based on what type of failure they have. 
                Classify them in the following format {'type': 'Offline-Module/Cell/Cell-Multi/Hot-Spot/Hot-Spot-Multi/Diode/Diode-Multi'}"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this infrared image of a PV module."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                seed=self.random_seed
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def process_images(self, folder_path, test_labels, classification_function):
        """Process images using the specified classification function."""
        data = []
        image_files = self.get_image_files(folder_path, test_labels)
        
        for filename in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(folder_path, filename)
            try:
                encoded_image = self.encode_image(image_path)
                content = classification_function(encoded_image)
                if content:  # Check if valid response
                    data.append({"image_name": filename, "content": content})
                    print(f"Processed {filename}: {content}")
                else:
                    print(f"Failed to get classification for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return pd.DataFrame(data)
    
    def evaluate_classification(self, df_results, df_metadata, label_map=None, output_file=None):
        """Evaluate classification results against metadata."""
        if df_results.empty or df_metadata is None:
            print("No results or metadata to evaluate")
            return None, 0.0
        
        # Merge results with metadata
        merged_df = pd.merge(df_metadata, df_results, on='image_name', how='inner')
        
        # Extract JSON content if needed
        def extract_type(text):
            try:
                if isinstance(text, str) and "type" in text:
                    # Try parsing as JSON first
                    try:
                        data = json.loads(text.replace("'", "\""))
                        return data["type"]
                    except json.JSONDecodeError:
                        # If not valid JSON, try extracting with string methods
                        if "type" in text and ":" in text:
                            return text.split("type")[1].split(":")[1].strip().strip("'\"}")
                return text
            except Exception:
                return text
                
        merged_df['content'] = merged_df['content'].apply(extract_type)
            
        # Process ground truth labels
        if label_map:
            try:
                labels = [label_map[f'{a}'] for a in merged_df['anomaly_class']]
                # Convert string labels to integers if necessary
                if all(isinstance(label, str) for label in labels):
                    labels = [int(x) if x.isdigit() else x for x in labels]
            except KeyError as e:
                print(f"Label mapping error: {e}")
                return merged_df, 0.0
        else:
            labels = merged_df['anomaly_class'].tolist()
        
        # Process model outputs
        outputs = merged_df['content'].tolist()
        if label_map:
            try:
                outputs = [label_map[f'{a}'] for a in outputs]
                # Convert string outputs to integers if necessary
                if all(isinstance(output, str) for output in outputs):
                    outputs = [int(x) if x.isdigit() else x for x in outputs]
            except KeyError as e:
                print(f"Output mapping error: {e}")
                return merged_df, 0.0
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, outputs)
        print(f"Classification accuracy: {accuracy:.4f}")
        
        # Generate more detailed metrics if classes are limited
        unique_labels = set(labels)
        if len(unique_labels) <= 10:  # Only generate detailed report for reasonable number of classes
            print("\nClassification Report:")
            print(classification_report(labels, outputs))
            
            # Save detailed results if output file is specified
            if output_file:
                merged_df['true_label'] = labels
                merged_df['predicted'] = outputs
                merged_df['correct'] = merged_df['true_label'] == merged_df['predicted']
                merged_df.to_csv(output_file, index=False)
                print(f"Detailed results saved to {output_file}")
        
        return merged_df, accuracy
    
    def run_binary_classification(self, dataset_file, image_folder, max_test_images):
        """Run binary classification workflow."""
        print("\n=== Running Binary Classification ===")
        dataset = self.load_yaml_dataset(dataset_file)
        if not dataset:
            return None, 0.0
            
        test_labels = dataset['test'][:max_test_images]
        
        # Process images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"binary_classification_results_{timestamp}.csv")
        
        results_df = self.process_images(image_folder, test_labels, self.get_binary_classification)
        results_df.to_csv(output_file, index=False)
        
        # Evaluate results
        metadata_df = self.load_metadata()
        detailed_output = os.path.join(DEFAULT_OUTPUT_DIR, f"binary_detailed_{timestamp}.csv")
        merged_df, accuracy = self.evaluate_classification(results_df, metadata_df, output_file=detailed_output)
        
        return merged_df, accuracy
    
    def run_detailed_classification(self, dataset_file, image_folder, max_test_images):
        """Run detailed multi-class classification workflow."""
        print("\n=== Running Detailed Classification ===")
        dataset = self.load_yaml_dataset(dataset_file)
        if not dataset:
            return None, 0.0
            
        test_labels = dataset['test'][:max_test_images]
        
        # Process images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"detailed_classification_results_{timestamp}.csv")
        
        results_df = self.process_images(image_folder, test_labels, self.get_detailed_classification)
        results_df.to_csv(output_file, index=False)
        
        # Evaluate results
        metadata_df = self.load_metadata()
        detailed_output = os.path.join(DEFAULT_OUTPUT_DIR, f"detailed_results_{timestamp}.csv")
        merged_df, accuracy = self.evaluate_classification(
            results_df, metadata_df, multi_learning_map, output_file=detailed_output
        )
        
        return merged_df, accuracy
    
    def run_reduced_classification(self, dataset_file, image_folder, max_test_images):
        """Run reduced classification workflow."""
        print("\n=== Running Reduced Classification ===")
        dataset = self.load_yaml_dataset(dataset_file)
        if not dataset:
            return None, 0.0
            
        test_labels = dataset['test'][:max_test_images]
        
        # Process images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"reduced_classification_results_{timestamp}.csv")
        
        results_df = self.process_images(image_folder, test_labels, self.get_reduced_classification)
        results_df.to_csv(output_file, index=False)
        
        # Evaluate results
        metadata_df = self.load_metadata()
        detailed_output = os.path.join(DEFAULT_OUTPUT_DIR, f"reduced_detailed_{timestamp}.csv")
        merged_df, accuracy = self.evaluate_classification(
            results_df, metadata_df, multi_reduction_learning_map, output_file=detailed_output
        )
        
        return merged_df, accuracy


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="PV Module Failure Classification using Vision AI")
    
    # Add arguments
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})")
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGE_FOLDER,
                        help=f"Path to image folder (default: {DEFAULT_IMAGE_FOLDER})")
    parser.add_argument('--metadata', type=str, default=DEFAULT_METADATA_FILE,
                        help=f"Path to metadata JSON file (default: {DEFAULT_METADATA_FILE})")
    parser.add_argument('--max-images', type=int, default=DEFAULT_MAX_TEST_IMAGES,
                        help=f"Maximum number of test images to process (default: {DEFAULT_MAX_TEST_IMAGES})")
    parser.add_argument('--mode', type=str, choices=['binary', 'detailed', 'reduced', 'all'], default='all',
                        help="Classification mode to run (default: all)")
    parser.add_argument('--binary-dataset', type=str, default='dataset_binary.yaml',
                        help="Path to binary classification dataset YAML")
    parser.add_argument('--multi-dataset', type=str, default='dataset_multi.yaml',
                        help="Path to multi-class classification dataset YAML")
    parser.add_argument('--reduced-dataset', type=str, default='dataset_classification_reduction.yaml',
                        help="Path to reduced classification dataset YAML")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print("PV Module Failure Classification")
    print("================================")
    print(f"Image folder: {args.images}")
    print(f"Metadata file: {args.metadata}")
    print(f"Max test images: {args.max_images}")
    print(f"Random seed: {args.seed}")
    print(f"Mode: {args.mode}")
    
    # Initialize classifier
    classifier = PVModuleClassifier(random_seed=args.seed)
    
    # Run selected classification mode(s)
    results = {}
    
    if args.mode in ['binary', 'all']:
        print("\nRunning binary classification...")
        binary_results, binary_accuracy = classifier.run_binary_classification(
            args.binary_dataset, args.images, args.max_images
        )
        results['binary'] = binary_accuracy
    
    if args.mode in ['detailed', 'all']:
        print("\nRunning detailed classification...")
        detailed_results, detailed_accuracy = classifier.run_detailed_classification(
            args.multi_dataset, args.images, args.max_images
        )
        results['detailed'] = detailed_accuracy
    
    if args.mode in ['reduced', 'all']:
        print("\nRunning reduced classification...")
        reduced_results, reduced_accuracy = classifier.run_reduced_classification(
            args.reduced_dataset, args.images, args.max_images
        )
        results['reduced'] = reduced_accuracy
    
    # Print summary of results
    print("\n=== Summary of Results ===")
    for mode, accuracy in results.items():
        print(f"{mode.capitalize()} Classification Accuracy: {accuracy:.4f}")
    
    print("\nResults saved in the 'results' directory.")


if __name__ == "__main__":
    main()