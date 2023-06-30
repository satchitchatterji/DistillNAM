# DistillNAM - CNN Knowledge Distillation using Neural Additive Models
---
_Satchit Chatterji, Gabriele Desimini, Marco Gallo_

**Note: This repository is a work in progress.**

With the surge of deep learning model usage, an important aspect of safety is understanding how a given model makes predictions. Here, we explore the use of neural additive models (NAMs) [1], which are interpretable by design, on non-tabular, image data (MNIST [2]). 

We also implement knowledge distillation, a method that is designed to assist the training of a `surrogate model’ by using the predictions of a pretrained ‘teacher model’. Specifically, we analyze the difference in predictions of NAMs with and without the assistance of a teacher CNN model.

---
Project undertaken for UvA MSc AI _Interpretability and Explainability in AI_ course 2023.
