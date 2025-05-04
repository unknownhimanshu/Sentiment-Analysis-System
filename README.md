# Sentiment-Analysis-System
A Python-based sentiment analysis application with a simple and intuitive GUI built using Tkinter. This project leverages machine learning and natural language processing (NLP) to classify user-input text as positive, negative, or neutral using a Logistic Regression model trained on labeled data.

🚀 Features
✅ Real-time sentiment prediction on user input

📊 Classification report display

🔤 Text feature extraction using TF-IDF

🧮 Model trained using Logistic Regression

🖥️ Simple and interactive GUI using Tkinter

📂 Dataset
The model is trained on a custom dataset (Book1.csv) which includes two columns:

text: The input text (e.g., tweets or reviews)

sentiment: The sentiment label (positive, negative, or neutral)

You can replace the dataset with any other labeled sentiment dataset of similar structure.

🛠️ Installation
Clone the repository

git clone https://github.com/your-username/sentiment-analysis-gui.git
cd sentiment-analysis-gui
Install the required packages

pip install -r requirements.txt
Add your dataset

Place your Book1.csv file in the root directory.

🧪 How to Run

python app.py
Once started, a window will appear where you can input text and analyze its sentiment.

📊 Libraries Used
Tkinter - For building the GUI

scikit-learn - For TF-IDF and Logistic Regression

pandas - For data handling

seaborn and matplotlib - For visualization (optional)

numpy

🧠 Model Performance
Accuracy: [Insert Accuracy]%

Evaluation using:

Classification Report

Confusion Matrix

📝 Future Improvements
Add support for deep learning models (e.g., LSTM)

Improve GUI design and user experience

Include a real-time feedback mechanism

🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

📃 License
This project is open-source and available under the MIT License.
