import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalysisGUI:
    def _init_(self, root):
        self.root = root
        self.root.title("Sentiment Analysis GUI")
        self.root.geometry("600x400")

        # Initialize components
        self.create_widgets()

    def create_widgets(self):
        # Text Entry
        text_entry_label = ttk.Label(self.root, text="Enter Text:")
        text_entry_label.pack(pady=10)
        self.text_entry = tk.Entry(self.root, width=50)
        self.text_entry.pack(pady=10)

        # Analyze Button
        analyze_button = ttk.Button(self.root, text="Analyze Sentiment", command=self.analyze_sentiment)
        analyze_button.pack(pady=10)

        # Result Label
        self.result_label = ttk.Label(self.root, text="")
        self.result_label.pack(pady=10)

        # Classification Report Label
        self.class_report_label = ttk.Label(self.root, text="")
        self.class_report_label.pack(pady=10)

    def analyze_sentiment(self):
        text_to_analyze = self.text_entry.get().strip()
        if not text_to_analyze:
            self.result_label.config(text="Please enter some text for analysis.")
            self.class_report_label.config(text="")
            return

        # Transform the input text into TF-IDF features
        text_tfidf = tfidf_vectorizer.transform([text_to_analyze])

        # Make predictions
        sentiment_prediction = model.predict(text_tfidf)

        # Display the result
        self.result_label.config(text=f"Predicted Sentiment: {sentiment_prediction[0]}")

        # Display classification report
        self.display_classification_report()

    def display_classification_report(self):
        # Make predictions on the test set
        y_pred = model.predict(X_test_tfidf)

        # Generate classification report
        class_report = classification_report(y_test, y_pred)

        # Display classification report in the GUI
        self.class_report_label.config(text="\nClassification Report:")
        self.class_report_label.config(text=class_report)

# Load the dataset using pandas
data1 = pd.read_csv(r"Book1.csv", encoding='latin1')

# Fill missing values with an empty string
data = data1.where((pd.notnull(data1)), '')

# Extract 'text' and 'sentiment' columns
data = data[['text', 'sentiment']]
data = data.dropna()

# Split the data into training and testing sets
X = data['text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create and train a Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Create GUI
if _name_ == "_main_":
    root = tk.Tk()
    app = SentimentAnalysisGUI(root)
    root.mainloop()


