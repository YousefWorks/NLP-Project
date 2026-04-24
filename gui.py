import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ----------------------------
# UI Settings
# ----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ----------------------------
# Download stopwords
# ----------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ----------------------------
# Load Models
# ----------------------------
try:
    model_nb = pickle.load(open("nb (1).pkl", "rb"))
    model_dt = pickle.load(open("dt.pkl", "rb"))
    model_svm = pickle.load(open("svm (1).pkl", "rb"))
    model_rf = pickle.load(open("rf (1).pkl", "rb"))
    model_nn = pickle.load(open("nn.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer (2).pkl", "rb"))
except Exception as e:
    print("Error loading models:", e)

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ----------------------------
# Prediction Function
# ----------------------------
def process_text():
    text = entry.get()

    if not text.strip():
        messagebox.showwarning("Empty Input", "Please enter a message!")
        return

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    preds = {
        "nb": model_nb.predict(vector)[0],
        "dt": model_dt.predict(vector)[0],
        "svm": model_svm.predict(vector)[0],
        "rf": model_rf.predict(vector)[0],
        "nn": model_nn.predict(vector)[0]
    }

    def update(lbl, val):
        if val == 1:
            lbl.configure(text="SPAM", text_color="#FF5252")
        else:
            lbl.configure(text="NOT SPAM", text_color="#69F0AE")

    update(res_nb, preds["nb"])
    update(res_dt, preds["dt"])
    update(res_svm, preds["svm"])
    update(res_rf, preds["rf"])
    update(res_nn, preds["nn"])

# ----------------------------
# GUI
# ----------------------------
app = ctk.CTk()
app.title("AI Spam Guardian")
app.geometry("600x650")

# Header
header = ctk.CTkFrame(app, fg_color="transparent")
header.pack(pady=25)

ctk.CTkLabel(header, text="📧 Spam Detector AI",
             font=ctk.CTkFont(size=28, weight="bold")).pack()

ctk.CTkLabel(header, text="Multi Model Classification System",
             text_color="gray").pack()

# Input
input_frame = ctk.CTkFrame(app, fg_color="transparent")
input_frame.pack(padx=30, pady=10, fill="x")

entry = ctk.CTkEntry(input_frame, placeholder_text="Enter message...",
                      height=45)
entry.pack(fill="x", pady=10)

ctk.CTkButton(input_frame, text="Analyze Message",
              command=process_text,
              height=45).pack(fill="x")

# Results
results = ctk.CTkFrame(app, corner_radius=15)
results.pack(padx=30, pady=20, fill="both", expand=True)

ctk.CTkLabel(results, text="Results",
             font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

def row(name):
    frame = ctk.CTkFrame(results, fg_color="transparent")
    frame.pack(fill="x", padx=20, pady=5)

    ctk.CTkLabel(frame, text=name,
                 font=ctk.CTkFont(size=14)).pack(side="left")

    lbl = ctk.CTkLabel(frame, text="Waiting...",
                       text_color="gray",
                       font=ctk.CTkFont(size=14, weight="bold"))
    lbl.pack(side="right")
    return lbl

res_nb = row("Naive Bayes")
res_dt = row("Decision Tree ")
res_svm = row("SVM")
res_rf = row("Random Forest")
res_nn = row("Neural Network ⭐")

# Footer
ctk.CTkLabel(app, text="Developed by Team",
             text_color="gray").pack(side="bottom", pady=10)

app.mainloop()