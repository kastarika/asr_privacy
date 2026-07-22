# AirSim RL Provenance Tracking System

Welcome to the AirSim RL Provenance Tracking project. This repository contains the code, models, and environments for training a drone to navigate complex geometries using Proximal Policy Optimization (PPO) in AirSim, alongside a robust provenance graph system for tracking hardware, code, and model dependencies.

## Architecture & Provenance Graph

The following Directed Acyclic Graph (DAG) illustrates the provenance tracking of our system. It tracks the exact lineage of code commits, training processes, and hardware execution environments to assist with root-cause analysis. For instance, you can use this to trace a simulated drone failure directly back to a specific code merge or hardware execution state.

```mermaid
graph TD
    %% Define Node Styling and Colors
    classDef actor fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px,color:#000;
    classDef process fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000;
    classDef artifact fill:#bbdefb,stroke:#0d47a1,stroke-width:2px,color:#000;
    classDef hardware fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef commit fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000;

    %% --- ACTORS ---
    Alice([Actor: Dev_Alice]):::actor
    CICD([Actor: CI_CD_Pipeline]):::actor

    %% --- HARDWARE ---
    GPU[Hardware: Training GPU A100]:::hardware
    CPU[Hardware: Edge CPU]:::hardware

    %% --- GIT HISTORY (The Timeline) ---
    CommitA[(Commit A: main baseline)]:::commit
    CommitB[(Commit B: feature-multi-rgb)]:::commit
    CommitC[(Commit C: PR Merge to main)]:::commit

    CommitA -->|PARENT_OF| CommitB
    CommitA -->|PARENT_OF| CommitC
    CommitB -->|PARENT_OF| CommitC

    %% --- ARTIFACTS & DATA ---
    Data[(Data: AirSim_Dataset_v1)]:::artifact
    EnvV2[Code: airsim_env.py v2_multi_rgb]:::artifact
    
    CommitB -.->|Generates| EnvV2

    %% --- TRAINING PROCESS ---
    Train[[Process: Training_Run_1]]:::process
    Model((Artifact: best_model_multi_rgb.zip)):::artifact

    Alice -->|WAS_ASSOCIATED_WITH| Train
    Data -->|USED| Train
    EnvV2 -->|USED| Train
    Train -->|EXECUTED_ON| GPU
    Train -->|WAS_GENERATED_BY| Model

    %% --- INFERENCE PROCESS (The Failure) ---
    Infer[[Process: Inference_Run_1]]:::process
    Telemetry((Artifact: Telemetry_Log_Failed)):::artifact

    CICD -->|WAS_ASSOCIATED_WITH| Infer
    CommitC -->|USED Code State| Infer
    Model -->|USED| Infer
    Infer -->|EXECUTED_ON| CPU
    Infer -->|WAS_GENERATED_BY| Telemetry
```


# AirSim RL Provenance Tracking System

Welcome to the AirSim RL Provenance Tracking project. This repository contains the code, models, and environments for training a drone to navigate complex geometries using Proximal Policy Optimization (PPO) in AirSim. 

## Hindsight Provenance Graph Architecture

The following Directed Acyclic Graph (DAG) illustrates the hindsight tracking of our system. It maps the exact lineage of code commits, training data, and hardware execution environments to assist with rapid root-cause analysis. 

To optimize for backward traversal during anomaly detection, all dependencies (Actors, Code, Data, and Hardware) are modeled as directed inputs flowing into the execution processes. If a failure occurs at the terminal evaluation node, the detection system simply follows the incoming edges backward to isolate the exact vulnerability.

```mermaid
graph TD
    %% Define Node Styling and Colors
    classDef actor fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px,color:#000;
    classDef process fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000;
    classDef artifact fill:#bbdefb,stroke:#0d47a1,stroke-width:2px,color:#000;
    classDef hardware fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef commit fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000;

    %% --- ACTORS & HARDWARE (The Inputs) ---
    Alice([Actor: Dev_Alice]):::actor
    CICD([Actor: CI_CD_Pipeline]):::actor
    GPU[Hardware: Training GPU A100]:::hardware
    CPU[Hardware: Edge CPU]:::hardware

    %% --- GIT HISTORY (The Timeline) ---
    CommitA[(Commit A: main baseline)]:::commit
    CommitB[(Commit B: feature-multi-rgb)]:::commit
    CommitC[(Commit C: PR Merge to main)]:::commit

    CommitA -->|PARENT_OF| CommitB
    CommitA -->|PARENT_OF| CommitC
    CommitB -->|PARENT_OF| CommitC

    %% --- ARTIFACTS & DATA (The Inputs) ---
    Data[(Data: AirSim_Dataset_v1)]:::artifact
    EnvV2[Code: airsim_env.py v2_multi_rgb]:::artifact
    CommitB -.->|GENERATES| EnvV2

    %% --- TRAINING PROCESS ---
    Train[[Process: Training_Run_1]]:::process
    Model((Artifact: best_model_multi_rgb.zip)):::artifact

    %% Everything flows INTO the process now for easy backtracking
    Alice -->|INITIATED| Train
    Data -->|USED_DATA| Train
    EnvV2 -->|USED_CODE| Train
    GPU -->|PROVIDED_COMPUTE| Train
    
    Train -->|GENERATED| Model

    %% --- INFERENCE PROCESS (The Failure) ---
    Infer[[Process: Inference_Run_1]]:::process
    Telemetry((Artifact: Telemetry_Log_Failed)):::artifact

    %% Everything flows INTO the inference now
    CICD -->|INITIATED| Infer
    CommitC -->|USED_CODE| Infer
    Model -->|USED_MODEL| Infer
    CPU -->|PROVIDED_COMPUTE| Infer
    
    Infer -->|GENERATED| Telemetry
```
