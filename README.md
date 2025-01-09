# aichatbot
# Large Language Model Chatbot Project

In this project, we are building a Large Language Model Chatbot focused on supervised and unsupervised learning knowledge. We used the pretrained model `Mistral-7B-Instruct-v0.2` as our initialization. We have obtained a final model with better performance in the focused field compared to the original one.

---

## Project Files and Descriptions

### 1. **modelTrainCode.ipynb**
**Purpose**: This Jupyter Notebook contains code to fine-tune a pretrained model `Mistral-7B-Instruct-v0.2`.

**Key Features**:
- Loads a base model (`Mistral-7B-Instruct-v0.2`).
- Configures the model for low-resource training with LoRA.
- Tokenizes and trains the model on prepared datasets.
- Saves the trained model for future use.

---

### 2. **TxT_transformation.ipynb**
**Purpose**: Prepares and reformats raw text data into a structured format suitable for model training.

**Key Features**:
- Reads and processes text data.
- Cleans and formats the data into meaningful paragraphs.
- Outputs text ready for Q&A generation and model training.

---

### 3. **formatted_qa_pairs.txt**
**Purpose**: Contains pre-generated and formatted question-answer pairs for supervised learning.

**Format**:
```plaintext
<s>[INS]@AI.<question>Q: {Question}</question>[/INS]<answer>A: {Answer}</answer></s>

**Usage**: Provide training data for the model to learn from structured Q&A pairs.
