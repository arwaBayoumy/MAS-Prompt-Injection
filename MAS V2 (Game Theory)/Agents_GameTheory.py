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
import re

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
            "LABEL_0": "Egyptian Arabic",    # 0
            "LABEL_1": "Levantine Arabic",   # 1
            "LABEL_2": "Tunisian Arabic",    # 2
            "Jordan": "Levantine Arabic",
            "Lebanon": "Levantine Arabic",
            "Palestine": "Levantine Arabic",
            "Tunis": "Tunisian Arabic",
            "Morocco": "Tunisian Arabic"
        }
        predicted_dialect = dialect_mapping.get(label, "Tunisian Arabic")
        print(f"Predicted Dialect: {predicted_dialect} with Confidence: {confidence}")
        return predicted_dialect, confidence

class ContextManipulationDetectionAgent:
    def __init__(self, csv_file_path='Agent2Book2.csv'):
        """
        Initialize the agent with the necessary data from the CSV file.
        """
        # Load the CSV file containing dangerous words and their categories
        self.df = pd.read_csv(csv_file_path)

        # Convert the data into useful structures
        self.dangerous_words = self.df['Word'].tolist()  # List of dangerous words
        self.categories = self.df['Category'].tolist()  # Corresponding categories
        
        # Define the severity weights for each category
        self.category_weights = {
            "Cybercrime": 3,
            "Misinformation": 2,
            "Hate Speech": 2,
            "Violence": 3,
            "Theft": 3,
            "Drug use & recklessness": 2,
        }

        # Define the available actions
        self.strategies = ['A0', 'A1', 'A2']  # A0: Ignore, A1: Flag, A2: Analyze Further
        
        # Prepare data for Game Theory
        self.prepare_game_data()

    def normalize_arabic(self, text: str) -> str:
        digit_map = {
            '٠': '0',
            '١': 'ا',
            '٢': 'ء',
            '٣': 'س',
            '٤': 'ع',
            '٥': 'ه',
            '٦': 'خ',
            '٩': 'و'
        }

        text = ''.join(digit_map.get(ch, ch) for ch in text)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # Remove Arabic diacritics
        norm_map = str.maketrans({
            'أ':'ا','إ':'ا','آ':'ا',
            'ؤ':'و','ئ':'ي','ة':'ه','ى':'ي'
        })
        return text.translate(norm_map)

    def prepare_game_data(self):
        """
        Prepare the data required for game theory. This includes calculating the weight for each word.
        """
        self.word_category_weights = []
        for word, category in zip(self.dangerous_words, self.categories):
            weight = self.category_weights.get(category, 1000)  # Default weight if category is unknown
            self.word_category_weights.append((str(word), weight))  # Ensure the word is a string
            
    def calculate_payoffs(self, manipulation_score):
        """
        Calculate the payoffs for each strategy based on the manipulation score.
        """
        # Payoff matrix: we define the payoffs for each strategy
        # The payoffs reflect how good each action is given the current manipulation score
        payoff_matrix = np.array([
            [0, -10, -5],  # Payoffs for Normal state (low risk)
            [-5, 5, 0],   # Payoffs for Potential Manipulation state (medium risk)
            [-10, 10, -5]  # Payoffs for Strong Manipulation state (high risk)
        ])

        if manipulation_score > 2:
            state = 2  # Strong Manipulation (S2)
        elif manipulation_score > 1:
            state = 1  # Potential Manipulation (S1)
        else:
            state = 0  # Normal (S0)

        # Calculate the expected payoff for each action
        payoffs = payoff_matrix[state]
        return payoffs

    def check_for_manipulation(self, prompt):
        """
        Check the given prompt for dangerous words and compute its manipulation score.
        """
        manipulation_score = 0
        updated_prompt = self.normalize_arabic(prompt)

        # Loop over the dangerous words and compute their impact on the manipulation score
        for dangerous_word, category in zip(self.dangerous_words, self.categories):
            if isinstance(dangerous_word, str) and dangerous_word.lower() in updated_prompt.lower():
                weight = self.category_weights.get(category, 1)  # Get the weight based on category
                manipulation_score += weight

        # Clamp the manipulation score to be between 0 and 5
        manipulation_score = min(max(manipulation_score, 0), 3)
        
        return manipulation_score

    def context_manipulation_detection(self, prompt, dialect):
        """
        The main function to detect context manipulation in a given prompt.
        """
        # Check for manipulation in the prompt
        manipulation_score = self.check_for_manipulation(prompt)

        # Calculate the payoffs for each strategy based on the manipulation score
        payoffs = self.calculate_payoffs(manipulation_score)

        # Choose the best action (the action with the highest payoff)
        best_action_index = np.argmax(payoffs)
        best_action = self.strategies[best_action_index]

        # Context analysis based on the manipulation score
        if manipulation_score > 3:
            context_analysis = f"Strong manipulation detected in {dialect} dialect."
        elif manipulation_score > 0:
            context_analysis = f"Potential manipulation detected in {dialect} dialect."
        else:
            context_analysis = f"No manipulation detected in {dialect} dialect."

        return context_analysis, manipulation_score, best_action

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
        normalized_context_score = context_score/3.0
        
        # Calculate the weighted sum to get the final risk score
        final_risk_score = (
            (dialect_score * self.weight_dialect) +
            (normalized_context_score * self.weight_context) +
            (response_score * self.weight_response)
        )
        
        # Determine the decision based on the final risk score
        if final_risk_score >= 0.6:
            decision = "Prompt Injection Detected"
        elif final_risk_score >= 0.48:
            decision = "Potential Prompt Injection Detected, Please Rephrase"
        else:
            decision = "Prompt Safe"
        
        return final_risk_score, decision
