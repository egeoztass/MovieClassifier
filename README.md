**Movie Review Sentiment Analysis with BERT**

This project is the culmination of my learning journey as a beginner in Natural Language Processing (NLP) and Machine Learning, specifically aiming to utilize the knowledge acquired from an NVIDIA certificate program focused on Transformer-Based Natural Language Processing Applications. The core of this project is to perform sentiment analysis on movie reviews using the BERT model, classifying them into positive or negative sentiments.

**About This Project**

As a novice in the NLP and deep learning domain, this project was undertaken to apply and consolidate the theoretical and practical knowledge gained during my NVIDIA certification. It marks my initial venture into applying transformers to address real-world challenges. The project, while a steep learning experience, has been incredibly fulfilling, showcasing the effectiveness of BERT models in text understanding and classification.

Despite achieving promising results, the model demonstrated slight overfitting—a learning curve in my journey. This experience has underscored the importance of model tuning and provided a clear direction for my continued learning in the field.

**Project Structure**

data/: Contains the dataset split into train.csv, val.csv, and test.csv.
model/: Directory for saving the trained model state dictionary and tokenizer.
src/: Source code including scripts for model training, evaluation, and inference.
README.md: This document, providing an overview and instructions for the project.
Setup

**Requirements**
Python 3.8+
PyTorch
Transformers
Pandas
Numpy
Install the necessary Python packages with:


pip install torch transformers pandas numpy

**Training the Model**

Place your dataset in the data/ directory, ensuring it's properly split.
From the src/ directory, run the training script:

python train_model.py

Training progress and accuracy metrics will be printed to the console.

**Evaluating the Model**

Evaluate the model on the validation and test datasets with:

python evaluate_model.py

This script loads the trained model and tokenizer, reporting accuracy metrics.

**Model Performance**

Our training process yielded the following accuracies over epochs, indicating the model's increasing proficiency in classifying movie reviews correctly:

**Epoch 1/4: Training Accuracy: 0.8858
Epoch 2/4: Training Accuracy: 0.9536
Epoch 3/4: Training Accuracy: 0.9857
Epoch 4/4: Training Accuracy: 0.9963**
On evaluating the model:

Validation Accuracy: **0.9305, 0.9281**
These results illustrate the model's strong performance, albeit with a hint of overfitting as evidenced by the discrepancy between training and validation accuracy.

**Reflections and Learning Outcomes**

This project has been a significant milestone in my journey towards mastering machine learning and NLP. Key learnings include:

Practical BERT Implementation: Understanding how to work with BERT for specific tasks.
Model Fine-tuning and Overfitting: The experience has highlighted the delicate balance between model complexity and its ability to generalize.
Documentation Importance: The process reinforced the value of thorough documentation for project clarity and future reference.
Future Directions

To address the slight overfitting observed, I plan to explore regularization techniques and data augmentation. Expanding my project portfolio with diverse NLP tasks and models will also be a priority.

**Contributions and License**

Feedback, contributions, and pull requests are welcome. Please feel free to open an issue or submit a pull request if you have suggestions or improvements.

This project is open-sourced under the MIT License.
