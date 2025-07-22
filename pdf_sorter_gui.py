# ファイル名: pdf_sorter_gui.py

import os
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

def sort_pdfs():
    download_dir = os.path.expanduser("~/Downloads")
    today_str = datetime.now().strftime("%Y-%m-%d")
    target_dir = os.path.join(download_dir, today_str)
    os.makedirs(target_dir, exist_ok=True)

    moved = []
    for filename in os.listdir(download_dir):
        if filename.lower().endswith(".pdf"):
            src = os.path.join(download_dir, filename)
            dst = os.path.join(target_dir, filename)
            try:
                shutil.move(src, dst)
                moved.append(filename)
            except Exception as e:
                print(f"Error: {e}")

    if moved:
        messagebox.showinfo("完了", f"{len(moved)} 個のPDFを {today_str} フォルダに移動しました。")
    else:
        messagebox.showinfo("完了", "移動するPDFはありませんでした。")

# GUIアプリ
root = tk.Tk()
root.title("PDF整理アプリ")
root.geometry("300x150")

label = tk.Label(root, text="PDFを日付ごとに整理します", pady=20)
label.pack()

btn = tk.Button(root, text="整理する", command=sort_pdfs, width=20, height=2)
btn.pack()

root.mainloop()