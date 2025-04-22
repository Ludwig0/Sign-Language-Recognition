## Introduction

We recognize that the rapid advances in deep learning today can greatly benefit special communities. By leveraging image recognition technology, we are able to identify and translate the sign language used by hearing-impaired individuals. In this project, we showcase a demo that detects and tracks hand keypoints.

## Development Process

We have adopted **Agile development** for this project:

1. **Rapid Prototyping**  
   Modern deep learning modules are highly encapsulated and allow us to spin up a working prototype very quickly—fully aligned with Agile principles.

2. **Iterative Improvement**  
   Achieving high accuracy with deep learning requires extensive training and tuning. A traditional Waterfall approach would demand a lengthy development cycle before any usable software emerges. With Agile, we first deliver a minimal working prototype and then iteratively refine its accuracy over successive sprints.

## Software Description

The overall architecture is divided into three modules, mirroring the intertwined "Requirements–Development–Testing" cycle of Agile:

- **`ui.py`**  
  Manages the user interface and overall look and feel of the application.

- **`model.py`**  
  Defines the neural network model, directly impacting recognition precision.

- **`architecture.py`**  
  Controls performance-related parameters, ensuring smooth real-time detection.

This modular design allows parallel development and supports incremental enhancements. While our current demo focuses solely on hand keypoint detection, the framework is readily extensible for full sign-language recognition in subsequent iterations.
