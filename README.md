# PAMAS: Self-Adaptive Multi-Agent System with Perspective Aggregation for Misinformation Detection

## 🧩 Overview
**PAMAS** is a **Perspective-Aware Multi-Agent System** designed for robust and interpretable **misinformation detection**.  
It organizes agents hierarchically and aggregates diverse perspectives through a self-adaptive mechanism, effectively mitigating the *information-drowning problem* and enhancing both efficiency and robustness.

## 🚀 Features
- **Hierarchical Agent Architecture** – Auditors, Coordinators, and a Decision-Maker cooperate for perspective-aware reasoning.  
- **Self-Adaptive Mechanisms** – Structural adaptation, targeted refinement, and confidence-guided routing.  
- **Perspective Aggregation** – Integrates multi-view analysis to highlight anomaly cues and suppress redundant signals.  
- **Plug-and-Play Design** – Easily extendable to other reasoning or detection tasks.

## 🛠️ How to Run

### 1) Preparation

#### Environment setup
Create and activate a Python environment, then install dependencies.

#### Download LLaMA (optional, if you run with local LLaMA)
Download **Llama-2-7b-hf** from Hugging Face:  
- https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main

Then place the model in this project **using the LLaMA naming format** (i.e., keep the directory name consistent with the official Hugging Face repo naming), so the code can locate it correctly.

#### Prepare API key
This project supports API calling as the primary backend (based on your configuration).  
Prepare your API key(s) according to your selected provider and configure them in your environment (or in the project config, depending on your setup).

---

### 2) Run
Simply run:
python main.py

## 🧠 Notes
We provide a complete runnable pipeline for Abnormal User Detection on the Amazon dataset:

We include the necessary files to reproduce one complete run.

If you want to train the system from scratch (instead of using the provided run-ready assets), you can retrain the model/pipeline accordingly.
The dataset is **Amazon** (For Abnormal User Detection).

