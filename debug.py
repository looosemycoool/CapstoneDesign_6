import os

manual_dir = r"C:\Users\yhm19\hybrid-rag-chatbot\data\manual_files"

for fname in os.listdir(manual_dir):
    fpath = os.path.join(manual_dir, fname)
    ext = os.path.splitext(fname)[1].lower()
    if ext == ".pdf":
        with open(fpath, "rb") as f:
            header = f.read(10)
        print(f"{fname}: {header}")