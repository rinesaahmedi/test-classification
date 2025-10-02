from datasets import load_dataset
import pandas as pd

dataset = load_dataset("ag_news")

print("Number of rows (train):", len(dataset["train"]))
print("Number of rows (test):", len(dataset["test"]))
print("Columns:", dataset["train"].column_names)

labels = dataset["train"].features["label"].names
print("Categories:", labels)

df = pd.DataFrame(dataset["train"])

print("\nSample counts per category:")
print(df["label"].value_counts().rename(index=lambda i: labels[i]))

print("\nSome sample texts:")
for i in range(5):
    print(f"Label: {labels[df.loc[i,'label']]}")
    print(f"Text: {df.loc[i,'text']}\n")
