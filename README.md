
# NER Amplifier 

## Introduction to Text Augmentation

Text augmentation is the process of creating synthetic textual data from existing data. It's a technique often used in Natural Language Processing (NLP) to enhance model performance, especially in situations where training data is limited. By introducing variations in the dataset, text augmentation helps in improving the robustness and generalization ability of NLP models.

## Project Overview

The NER Amplifier project provides a specialized approach to text augmentation focusing on Entity Recognition tasks. Currently, the NER Amplifier project includes an implementation for noun augmentation and we are actively working on developing and integrating other techniques, such as entity swapping-removal-replacement-insertion, as well as other token and character level augmentations. These enhancements aim to offer a comprehensive suite of tools for robust and effective data augmentation in NER tasks, ultimately contributing to the creation of more versatile and resilient NER models.


## Installation Guide

To get started with this project, you need to have Python >= 3.10 and Poetry installed on your system. Here's a quick guide on setting it up:

1. **Clone the Repository**: Clone this repository to your local machine.
   ```bash
   git clone https://github.com/PanosBn/ner-amplifier

2. **Install Dependencies**: Navigate to the project directory and install with.
   ```bash
   poetry install 
   ```

3. **Sense2Vec and Word2Vec Models**: Download the pre-trained Sense2Vec and Word2Vec models as required and place them in an accessible directory.

## Usage Example

Here's a simple example of how to use the Noun Augmenter in your Python script:

```python
from amplifier import Corpus, NounAugmenter

# Define the column mapping for your dataset
column_mapping = {"word": 0, "ner": 3}

# Load your dataset
corpus = Corpus(file_path="path_to_your_dataset.txt", column_mapping=column_mapping)

# Initialize the augmenter
augmenter = NounAugmenter()

# Apply augmentation using Sense2Vec
augmenter.noun_augment_sense2vec(corpus, model_path="path_to_sense2vec_model")

# Export the augmented corpus to a file
corpus.export_to_conll("augmented_corpus.txt", delimiter='\t')
```

