from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import os

# Agent 1 (Dialectal Analysis)
class DialectalAnalysisAgent:
    def __init__(self, model_name="CAMeL-Lab/bert-base-arabic-camelbert-msa-did-madar-twitter5"):
        # Initialize the model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def preprocess_data(self, train_file='train.csv', test_file='test.csv'):
        # Load datasets
        self.train_dataset = pd.read_csv(train_file)
        self.test_dataset = pd.read_csv(test_file)

        # Optionally, check the splits
        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Testing dataset size: {len(self.test_dataset)}")

        # Load the dataset
        dataset = load_dataset('csv', data_files={'train': train_file, 'test': test_file}, delimiter=',')

        # Preprocess the dataset
        def preprocess_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

        # Apply preprocessing
        self.train_dataset = dataset['train'].map(preprocess_function, batched=True)
        self.test_dataset = dataset['test'].map(preprocess_function, batched=True)

    def train_model(self):
        # Set training arguments
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # number of training epochs
            per_device_train_batch_size=8,   # batch size for training
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,                         # the model to be trained
            args=training_args,                  # training arguments
            train_dataset=self.train_dataset,         # training dataset
            eval_dataset=self.test_dataset            # evaluation dataset
        )

        # Fine-tune the model
        trainer.train()

        # Save the model after training
        self.model.save_pretrained("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")

    def load_trained_model(self):
        # Load the fine-tuned model
        self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
        self.tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

        # Load the classification pipeline
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def predict_dialect(self, text):
        # Test the function with a new example
        result = self.classifier(text)

        # Display the result
        label = result[0]['label']
        confidence = result[0]['score']

        # Map the label to the corresponding dialect
        dialect_mapping = {
            "Egypt": "Egyptian Arabic",
            "Jordan": "Levantine Arabic",
            "Lebanon": "Levantine Arabic",
            "Palestine": "Levantine Arabic",
            "Tunis": "Tunisian Arabic",
            "Morocco": "Tunisian Arabic"
        }

        predicted_dialect = dialect_mapping.get(label, "Unknown")
        print(f"Predicted Dialect: {predicted_dialect} with Confidence: {confidence}")
        return predicted_dialect, confidence

# Agent 2 (Context Manipulation Detection)
import pandas as pd
import numpy as np
import re

class ContextManipulationDetectionAgent:
    def __init__(self, csv_file_path='Agent2Book.csv'):
        """
        Initialize the agent with the necessary data from the CSV file.
        """
        # Load the CSV file containing dangerous words, weights, and prefixes
        self.df = pd.read_csv(csv_file_path)

        # Convert the data into useful structures
        self.dangerous_words = self.df['Dangerous Word'].tolist()
        self.weights = self.df['Weight'].tolist()
        self.prefixes = self.df['Prefixes'].fillna('').tolist()

        # Define states and actions
        self.states = ['S0', 'S1', 'S2']  # S0: Normal, S1: Potential Manipulation, S2: Strong Manipulation
        self.actions = ['A0', 'A1', 'A2']  # A0: Ignore, A1: Flag, A2: Analyze Further

        # Define the MDP Bellman Equation rewards and transition probabilities
        self.payoff_matrix = np.array([
            [0, -10, -5],   # S0: Normal
            [-5, 10, 7],    # S1: Potential Manipulation
            [-10, 20, 15]   # S2: Strong Manipulation
        ])

        self.T = np.array([
            [0.8, 0.15, 0.05],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])

    def normalize_arabic(self, text: str) -> str:
        digit_map = {
            '0':'0','٠':'0','1':'ا','١':'ا','2':'ء','٢':'ء',
            '3':'ع','٣':'ع','4':'ش','٤':'ش','5':'خ','٥':'خ',
            '6':'ط','٦':'ط','7':'ح','٧':'ح','8':'غ','٨':'غ',
            '9':'ظ','٩':'ظ'
        }
        text = ''.join(digit_map.get(ch, ch) for ch in text)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        norm_map = str.maketrans({
            'أ':'ا','إ':'ا','آ':'ا',
            'ؤ':'و','ئ':'ي','ة':'ه','ى':'ي'
        })
        return text.translate(norm_map)

    def check_prefix_in_prompt(self, prompt, dangerous_word, prefixes):
        prompt_lower = prompt.lower()
        dangerous_word_lower = dangerous_word.lower()

        for prefix in prefixes:
            prefix = prefix.strip()
            if prefix == '':
                continue
            if prefix.lower() in prompt_lower and dangerous_word_lower in prompt_lower:
                prefix_pos = prompt_lower.find(prefix.lower())
                word_pos = prompt_lower.find(dangerous_word_lower)
                if prefix_pos < word_pos:
                    return True
        return False

    def bellman_eq(self, state, action, manipulation_score):
        action_index = self.actions.index(action)
        reward = self.payoff_matrix[state, action_index] + manipulation_score
        return reward

    def context_manipulation_detection(self, prompt, dialect):
        manipulation_score = 0

        for dangerous_word, weight, prefix in zip(self.dangerous_words, self.weights, self.prefixes):
            word_weight = weight
            if self.check_prefix_in_prompt(prompt, dangerous_word, prefix.split(',')):
                word_weight += 3
            if dangerous_word.lower() in prompt.lower():
                manipulation_score += word_weight

        if manipulation_score > 4:
            state = 2
        elif manipulation_score > 3:
            state = 2
        elif manipulation_score > 0:
            state = 1
        else:
            state = 0
        print(f"manipulation_score: {manipulation_score}")

        rewards = [self.bellman_eq(state, action, manipulation_score) for action in self.actions]
        best_action_index = np.argmax(rewards)
        action = self.actions[best_action_index]

        context_analysis = f"Prompt analysis for {dialect} dialect: "
        if state == 2:
            context_analysis += "Strong manipulation detected."
        elif state == 1:
            context_analysis += "Potential manipulation detected."
        else:
            context_analysis += "No manipulation detected."

        return context_analysis, manipulation_score, action

# Agent 3 (Response Monitoring)
class ResponseMonitoringAgent:
    def __init__(self, csv_file_path='Results.xlsx'):
        """
        Initialize the agent with the necessary data and setup.
        """
        # Load the dataset
        if not os.path.exists(csv_file_path):
            print(f"ERROR: Dataset file {csv_file_path} not found.")
            raise SystemExit

        self.df = pd.read_excel(csv_file_path)

        if 'Prompt' not in self.df.columns or 'Model Output' not in self.df.columns or 'Result' not in self.df.columns:
            print("ERROR: Dataset missing required columns.")
            raise SystemExit

        # Split Result into MainCategory and SubCategory
        self.df[['MainCategory', 'SubCategory']] = self.df['Result'].str.split(', ', expand=True)
        self.df = self.df.dropna(subset=['Prompt', 'Model Output', 'SubCategory'])

        # Combine Prompt + Model Output
        self.df['Combined'] = self.df['Prompt'].astype(str) + " || " + self.df['Model Output'].astype(str)

        # Encode labels (SubCategory)
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df['SubCategory'])

        # Create pipeline and train the model
        self.nb_model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            MultinomialNB()
        )
        self.nb_model.fit(self.df['Combined'], self.y)

        # Category mapping to MainCategory
        self.subcategory_to_main = {
            "Advice": "Refusal",
            "Direct": "Refusal",
            "Unsafe": "Non-Refusal",
            "Misunderstanding": "Non-Refusal"
        }

        print("Agent 3 (Naive Bayes with prompt+output) trained successfully.")

    def classify_llm_output(self, prompt, model_output):
        """
        Classify the LLM output based on the prompt and model output using the trained model.
        """
        combined = prompt + " || " + model_output
        label_id = self.nb_model.predict([combined])[0]
        label = self.label_encoder.inverse_transform([label_id])[0]
        proba = self.nb_model.predict_proba([combined])[0]
        proba_dict = {
            self.label_encoder.inverse_transform([i])[0]: round(float(p), 3)
            for i, p in enumerate(proba)
        }
        main_category = self.subcategory_to_main.get(label, "Unknown")
        
        return {
            "Main Category": main_category,
            "Predicted Category": label,
            "Confidence Scores": proba_dict,
            "Conf": proba_dict.get(label)
        }

# Agent 4 (Decision Fusion)
class DecisionFusionAgent:
    def __init__(self, weight_dialect=0.10, weight_context=0.25, weight_response=0.65):
        self.weight_dialect = weight_dialect
        self.weight_context = weight_context
        self.weight_response = weight_response

    def decision_fusion(self, dialect_score, context_score, response_score):
        # Check types to ensure we have numeric values
        if not isinstance(context_score, (int, float)):
            raise ValueError(f"Expected numeric value for context_score, got {type(context_score)}")
        if not isinstance(response_score, (int, float)):
            raise ValueError(f"Expected numeric value for response_score, got {type(response_score)}")

        # Normalize the context and response scores to be between 0 and 1
        normalized_context_score = max(0, min(context_score / 4.0, 1))

        
        # Calculate the weighted sum to get the final risk score
        final_risk_score = (
            (dialect_score * self.weight_dialect) +
            (normalized_context_score * self.weight_context) +
            (response_score * self.weight_response)
        )
        
        # Determine the decision based on the final risk score
        if final_risk_score >= 0.60:
            decision = "Prompt Injection Detected"
        elif final_risk_score >= 0.48:
            decision = "Potential Prompt Injection Detected, Please Rephrase"
        else:
            decision = "Prompt Safe"
        
        return final_risk_score, decision
