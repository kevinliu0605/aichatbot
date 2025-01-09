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
```

**Usage**: Provide training data for the model to learn from structured Q&A pairs.

### 4. **Validation_Data.txt**
**Purpose**: Contains validation data in the form of multiple-choice questions.

**Format**:
```plaintext
<s>[INST]@AI_Supervised_Unsupervised.{Question}. [/INST]{Options}</s>{Correct Answer}
```

**Usage**: Evaluate the model's performance on unseen data to ensure generalization.

### 5. **Human Evaluation of Performance.docx**
**Purpose**: Documents the human evaluation process and results of the model's performance.

**Key Contents**:
- Comparison of model performance before and after training.
- Sample questions and detailed answers illustrating improvements.

---

### 6. **Gemini_Data_Generation.ipynb**
**Purpose**: Generates additional Q&A pairs using the Google Gemini 1.5 Flash API.

**Key Features**:
- Connects to the Gemini API with an API key.
- Uses system prompts to generate questions and answers from input text.
- Outputs data in the desired format for model training.

---

### 7 & 8. **adaptor.config.json** & **model.safetensors**
**Purpose**:
1. **adaptor.config.json**: Stores basic information about the parameters used during training.
2. **model.safetensors**: Stores all parameter data that the model has been trained on.

**Key Contents**:
- **adaptor.config.json**: Contains metadata on LoRA parameters, base model configurations, and other training settings.
- **model.safetensors**: Stores all checkpoints in dictionary form.

---

### 9. **unsupervised_reformat.txt**
**Purpose**: Contains cleaned and reformatted text related to unsupervised learning, sourced from the MIT Chapter 12 textbook.

**Use**:
- Serves as input for generating Q&A pairs using the Gemini API.
- Provides source material for unsupervised learning tasks.

---

### 10. **577_Final_Report.pdf**
**Purpose**: Contains the final report of this project in LaTeX format.

---

## How to Use the Files

### Generate Data
- Use `Gemini_Data_Generation.ipynb` to generate additional Q&A pairs, especially for unsupervised topics.
- Input text can be sourced from `unsupervised_reformat.txt`.

### Prepare Data
- Process raw text with `TxT_transformation.ipynb` to create structured data.
- Combine generated data with `formatted_qa_pairs.txt` for training.

### Train the Model
- Train the model using `modelTrainCode.ipynb`.
- Ensure datasets are prepared and tokenized before training.
- Save the trained model to `adaptor.config.json` and `model.safetensors`.

### Validate and Evaluate
- Use `Validation_Data.txt` for model validation.
- Document evaluation results in `Human Evaluation of Performance.docx`.

---

**Contributors**: Hanwen Liu 
**License**: Stony Brook University @ 2024