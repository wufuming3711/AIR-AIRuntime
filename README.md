# AIR-AIRuntime

## Framework Introduction

AIR-AIRuntime is an efficient and flexible deep learning inference framework designed based on the workflow pattern. It allows users to easily construct, manage, and optimize complex algorithm processes through configuration files. The framework not only supports parallel execution of multiple algorithms but also leverages multi-GPU for high-performance computing, thereby significantly improving processing speed.

### Key Features

- **Workflow-Based Design**: AIR-AIRuntime supports building algorithm schemes in a stream manner, meaning that different processing steps (such as preprocessing, model inference, post-processing, etc.) can be linked together to form a complete data processing pipeline.
- **Configuration File Driven**: Using the `config_workflow.json` configuration file to specify different algorithm schemes and set corresponding parameters makes it easy and quick to switch and adjust algorithm schemes.
- **Support for Parallel Multiple Algorithms**: The framework design considers the needs of multitasking, allowing multiple algorithms or models to run simultaneously, with each task being independently configured and optimized.
- **Multi-GPU Parallel Inference**: For tasks requiring substantial computational resources, AIR-AIRuntime can distribute tasks across multiple GPUs for parallel processing, greatly enhancing processing efficiency.
- **TensorRT Optimized Deployment**: Supports using NVIDIA TensorRT for optimizing the deployment of single detection algorithms and classification algorithms, as well as supporting the serial deployment of detection and classification algorithms to achieve higher inference speeds and lower latency.

### Technical Details

- **Configuration Management**: All algorithms and process configurations are managed through the `config_workflow.json` file. This file defines the components within the workflow and their interrelationships.
- **Model Deployment**: In addition to standard model formats, AIR-AIRuntime also supports TensorRT-formatted models, which helps achieve more efficient performance in production environments.
- **Extensibility**: The framework was designed with extensibility in mind, allowing users to add new modules or modify existing ones to adapt to different application scenarios.

### Application Scenarios

AIR-AIRuntime is particularly suitable for applications that require rapid iteration of algorithm schemes and the processing of large-scale datasets, including but not limited to:

- **Autonomous Driving**: Real-time processing of large amounts of sensor data in vehicle environmental perception systems, such as image recognition and obstacle detection.
- **Medical Imaging Analysis**: Rapid and accurate analysis of medical images to assist doctors in making diagnoses.
- **Intelligent Security**: Face recognition, behavior analysis, and other functions in video surveillance systems.
- **Retail Analytics**: Big data processing for customer behavior analysis and product recommendations.

### Developer Guide

To help developers make better use of AIR-AIRuntime, we provide a series of documentation and support resources, including but not limited to:

- **Quick Start Guide**: A comprehensive tutorial from setting up the environment to running your first example program.
- **API Documentation**: Detailed descriptions of all interfaces provided by the framework and their usage methods.
- **Case Studies**: Real-world examples showing how to use AIR-AIRuntime to solve practical problems.
- **Community Support**: Join our developer community to exchange experiences with other users and get technical support.
