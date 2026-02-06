# Construction Site Safety Detector (.NET)

**The same hard hat detection model rebuilt with the .NET stack - ML.NET, ASP.NET Razor Pages, and C#.**

[![.NET](https://img.shields.io/badge/.NET-9.0-purple)](https://dotnet.microsoft.com/)
[![ML.NET](https://img.shields.io/badge/ML.NET-5.0-blue)](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)
[![ASP.NET](https://img.shields.io/badge/ASP.NET-Razor_Pages-green)](https://learn.microsoft.com/aspnet/core/razor-pages)
[![Accuracy](https://img.shields.io/badge/Val_Accuracy-91.8%25-brightgreen)](#-results)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> This is the **.NET version** of my [hard-hat-detector](https://github.com/kieranpcremin/hard-hat-detector) project (Python/PyTorch). Same problem, different stack.

---

## 🎯 What This Project Does

Upload an image of a construction worker and the model classifies whether they are wearing a hard hat or not, with a confidence score.

- **Binary classification** - Hard Hat vs No Hard Hat
- **Transfer learning** - ResNet V2 50 pre-trained on ImageNet, retrained via ML.NET
- **Web demo** - ASP.NET Razor Pages app for real-time predictions
- **Thread-safe inference** - `PredictionEnginePool` for concurrent web requests

---

## 🖥️ Demo

The Razor Pages web app lets you upload any image and get an instant prediction with confidence scores.

```bash
dotnet run --project SafetyDetector.Web
```

<!-- ![Demo Screenshot](assets/demo_screenshot.png) -->

---

## ✨ How It Works

```
Input Image (JPEG/PNG)
       |
       v
  ┌─────────────────────────────────────┐
  │  LoadRawImageBytes                  │
  │  Read file into byte[]             │
  └─────────────────────────────────────┘
       |
       v
  ┌─────────────────────────────────────┐
  │  ResNet V2 50 Backbone              │
  │  Bottleneck feature extraction      │
  │  (frozen, pre-trained on ImageNet)  │
  └─────────────────────────────────────┘
       |
       v
  ┌─────────────────────────────────────┐
  │  Retrained Final Layer              │
  │  Custom classifier for 2 classes    │
  └─────────────────────────────────────┘
       |
       v
  "Hard Hat" or "No Hard Hat" + confidence %
```

### Training Strategy

ML.NET's `ImageClassificationTrainer` uses a two-phase approach:

1. **Bottleneck computation** - The frozen ResNet V2 50 backbone processes every image and caches the output features to disk. This only needs to happen once.
2. **Final layer retraining** - Only the classification layer is trained on the cached features using gradient descent over 15 epochs.

> Unlike PyTorch where every epoch re-runs images through the entire network, ML.NET caches the frozen backbone outputs once and trains only on cached features - making training **5x faster**.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **91.8%** |
| **Architecture** | ResNet V2 50 (25.6M total params) |
| **Trainable Parameters** | Final classification layer only |
| **Training Time** | ~2.9 min (CPU) |
| **Epochs** | 15 |
| **Model File Size** | ~91 MB (.zip) |

### Confusion Matrix

|  | Predicted Hard Hat | Predicted No Hard Hat | Recall |
|--|---|---|---|
| **Actual Hard Hat** | 71 | 8 | 89.9% |
| **Actual No Hard Hat** | 6 | 85 | 93.4% |

The model is slightly better at detecting "no hard hat" (93.4%) than "hard hat" (89.9%) - it's more likely to miss a hard hat than to falsely detect one.

---

## 🔬 Python vs .NET - Framework Comparison

Both models were trained on the same 919-image dataset (500 hard hat + 419 no hard hat).

### Head-to-Head Results

| Metric | Python (PyTorch) | .NET (ML.NET) |
|--------|-----------------|---------------|
| **Validation Accuracy** | **94.6%** | **91.8%** |
| **Training Time** | ~15 min (CPU) | ~2.9 min (CPU) |
| **Model File Size** | ~45 MB (.pth) | ~91 MB (.zip) |
| **Backbone** | ResNet18 (11.7M params) | ResNet V2 50 (25.6M params) |
| **Fine-tuning** | ✅ Freeze + unfreeze last 2 blocks | ❌ Final layer only |
| **Code Volume** | ~400 lines across 4 files | ~200 lines across 3 files |

### Why the 3% Accuracy Gap?

```
Python (PyTorch):
  Phase 1: Freeze backbone, train classifier     → ~89% val accuracy
  Phase 2: Unfreeze last 2 blocks, fine-tune      → ~94.6% val accuracy
                                                     ↑ +5.6% from fine-tuning

.NET (ML.NET):
  Only phase: Freeze backbone, train classifier   → ~91.8% val accuracy
  No fine-tuning available                         → That's the ceiling
```

ML.NET's `ImageClassificationTrainer` only supports retraining the final classification layer. It does **not** support unfreezing and fine-tuning deeper layers. However, it starts with a more powerful backbone (ResNet V2 50 vs ResNet18), which partly compensates.

### 🔴 The Yellow Hair Test - Accuracy Doesn't Tell the Full Story

We tested both models with an image of a person with big yellow hair (no hard hat):

| Test | Python (ResNet18) | .NET (ResNet V2 50) |
|------|-------------------|---------------------|
| **Yellow hair image** | ❌ **WRONG** - "Hard Hat" detected | ✅ **CORRECT** - "No Hard Hat" (70% confidence) |
| **Overall val accuracy** | 94.6% | 91.8% |

The Python model with **higher** overall accuracy was fooled, while the .NET model with **lower** accuracy got it right.

**Why?** ResNet V2 50 (50 layers) builds more complex feature representations than ResNet18 (18 layers). With more depth, it can distinguish "yellow blob on head" from "actual hard hat with specific shape, material, and structure." ResNet18 learned a simpler shortcut: bright color near head = hard hat.

> **The lesson:** A single accuracy number doesn't capture model robustness. A model can score higher on a test set but still be more brittle in the real world. Testing with adversarial/edge cases reveals the real quality of what the model learned.

### Framework Trade-offs

| Aspect | PyTorch Wins | ML.NET Wins |
|--------|-------------|-------------|
| **Control** | ✅ Full access to every weight, gradient, layer | - |
| **Learning** | ✅ Forces you to understand the math | - |
| **Speed to prototype** | - | ✅ Much less code needed |
| **Enterprise integration** | - | ✅ Native .NET, NuGet, DI, ASP.NET |
| **Thread safety** | - | ✅ `PredictionEnginePool` built-in |
| **Architecture choices** | ✅ Any model from anywhere | 4 pre-built options |
| **Fine-tuning** | ✅ Freeze/unfreeze any layer | Final layer retraining only |
| **Training speed** | - | ✅ 5x faster with bottleneck caching |

**Key Insight:** ML.NET is more like a **tool** - point it at data, get a model. PyTorch is more like a **workshop** - you build the model yourself. Both have their place.

---

## 📁 Project Structure

```
safety-detector-net/
├── SafetyDetector.sln                  # Solution file
├── SafetyDetector.Shared/              # Shared class library
│   └── DataModels/
│       ├── ImageData.cs                # Image path + label (disk loading)
│       ├── ModelInput.cs               # byte[] image + encoded label (trainer input)
│       └── ModelOutput.cs              # Predicted label + score[] (prediction output)
├── SafetyDetector.Training/            # Console app - model training
│   └── Program.cs                      # Full ML.NET training pipeline
├── SafetyDetector.Web/                 # Razor Pages - web demo
│   ├── Program.cs                      # DI with PredictionEnginePool
│   ├── Pages/
│   │   ├── Index.cshtml                # Upload form + results UI
│   │   └── Index.cshtml.cs             # Prediction handler
│   └── MLModel/                        # Trained model goes here
├── data/                               # Training images (not in repo)
└── README.md
```

### Why 3 Projects?

| Project | Responsibility | Why Separate? |
|---------|---------------|---------------|
| **Shared** | Data model classes (ImageData, ModelInput, ModelOutput) | Both Training and Web need these - avoids duplication |
| **Training** | ML.NET training pipeline | Console app, run once, produces model.zip |
| **Web** | Razor Pages demo | References model.zip, serves predictions |

---

## 🚀 Setup

### 1. Clone & Restore

```bash
git clone https://github.com/kieranpcremin/hard-hat-detector-ml-dot-net.git
cd hard-hat-detector-ml-dot-net
dotnet restore
```

### 2. Download Dataset

Use the same dataset as the [Python version](https://github.com/kieranpcremin/hard-hat-detector). Organize into a flat folder-per-class structure:

```
data/
├── hard_hat/        # ~500 JPEG/PNG images
│   ├── img001.jpg
│   └── ...
└── no_hard_hat/     # ~400 JPEG/PNG images
    ├── img001.jpg
    └── ...
```

**Dataset sources:**
- [Kaggle - Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- [Roboflow - Hard Hat Classification](https://universe.roboflow.com/search?q=hard+hat+classification)

> **Note:** ML.NET handles train/validation splitting automatically (80/20) - no need to pre-split into train/val folders like in the Python version.

### 3. Train the Model

```bash
dotnet run --project SafetyDetector.Training
```

This will:
- Load all images from `data/` (folder name = class label)
- Download pre-trained ResNet V2 50 weights (~100MB, first run only)
- Compute and cache bottleneck features using the frozen backbone
- Train the final classification layer for 15 epochs
- Evaluate on a 20% validation split
- Save model to `SafetyDetector.Web/MLModel/model.zip`

> **First run** is slower due to weight download and bottleneck computation. Subsequent runs reuse cached features and are much faster.

### 4. Run the Web App

```bash
dotnet run --project SafetyDetector.Web
```

Open the URL shown in the terminal (typically `http://localhost:5236`). Upload an image to get a prediction.

---

## 🛠️ Tech Stack

- **ML.NET 5.0** - Microsoft's machine learning framework for .NET
- **TensorFlow** (via SciSharp.TensorFlow.Redist 2.16.0) - Backend for transfer learning
- **ASP.NET Core 9.0 Razor Pages** - Server-rendered web UI
- **Bootstrap 5** - CSS framework for responsive design
- **PredictionEnginePool** - Thread-safe model inference for web apps

---

## ⚠️ Key Gotchas

| Issue | Details |
|-------|---------|
| **TensorFlow version** | Must use `SciSharp.TensorFlow.Redist` **2.16.0** for ML.NET 5.0. Older versions (2.3.1) crash with `TF_StringInit` not found |
| **Image formats** | Only JPEG and PNG supported by `LoadRawImageBytes` |
| **Model file size** | Output `.zip` is ~91MB (contains full TensorFlow graph + retrained weights) |
| **First run download** | Pre-trained ResNet weights download automatically on first training run |
| **CPU only by default** | GPU requires specific CUDA version + separate NuGet GPU package |
| **No fine-tuning** | ML.NET only retrains the final layer - no unfreezing of deeper layers |

---

## 🧠 Key Concepts Demonstrated

| Concept | Where | What I Learned |
|---------|-------|----------------|
| **Transfer Learning** | `Program.cs` (Training) | Using ImageNet pre-trained ResNet V2 50 as a feature extractor |
| **ML.NET Pipelines** | `Program.cs` (Training) | MapValueToKey, LoadRawImageBytes, ImageClassificationTrainer chain |
| **Bottleneck Caching** | Training pipeline | Frozen backbone outputs cached to disk - much faster than re-running every epoch |
| **PredictionEnginePool** | `Program.cs` (Web) | Thread-safe inference pattern for ASP.NET apps - `PredictionEngine` itself is NOT thread-safe |
| **Model Serialization** | Training → Web | Self-contained `.zip` with TF graph + weights + pipeline metadata |
| **Razor Pages** | Web app | Clean server-side rendered UI with form handling |

---

## 👨‍💻 Author

**Kieran Cremin**
Built with assistance from Claude (Anthropic)

---

## 📄 License

MIT License - Free to use, modify, and distribute.
