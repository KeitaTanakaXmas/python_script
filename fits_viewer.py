import tkinter as tk
from tkinter import filedialog, messagebox
from astropy.io import fits
import numpy as np

# ファイル選択ダイアログでFITSファイルを選ぶ
def open_fits_file():
    file_path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits.gz;*.fits")])
    if file_path:
        try:
            hdulist = fits.open(file_path)
            display_header(hdulist)
            display_hdu_list(hdulist)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open FITS file: {e}")

# ヘッダー情報を表示する
def display_header(hdulist):
    header_text.delete(1.0, tk.END)  # テキストボックスをクリア
    header = hdulist[0].header  # 最初のHDUのヘッダー
    for keyword, value in header.items():
        header_text.insert(tk.END, f"{keyword}: {value}\n")

# HDUのリストを表示
def display_hdu_list(hdulist):
    hdu_listbox.delete(0, tk.END)  # リストをクリア
    for i, hdu in enumerate(hdulist):
        hdu_listbox.insert(tk.END, f"HDU {i}: {hdu.name}")

# HDUが選択されたときにデータを表示する
def on_hdu_select(event):
    selection = hdu_listbox.curselection()
    if selection:
        hdu_index = selection[0]
        hdu = hdulist[hdu_index]
        display_data(hdu, hdu_index)

# HDUのデータを表示
def display_data(hdu, hdu_index):
    data_text.delete(1.0, tk.END)  # テキストをクリア
    data = hdu.data
    data_text.insert(tk.END, f"HDU {hdu_index} Data:\n")
    
    if data is not None:
        if 'ENERGY' in data.dtype.names:
            energy_data = data['ENERGY']
            data_text.insert(tk.END, f"Energy Data (First 5 values): {energy_data[:5]}...\n")
            
            # 統計情報
            stats = f"Max: {np.max(energy_data):.2f}, Min: {np.min(energy_data):.2f}, " \
                    f"Mean: {np.mean(energy_data):.2f}, Std: {np.std(energy_data):.2f}"
            data_text.insert(tk.END, stats + "\n")
        else:
            data_text.insert(tk.END, "No 'ENERGY' field found in data.\n")
    else:
        data_text.insert(tk.END, "No data found in this HDU.\n")

# メインウィンドウの設定
root = tk.Tk()
root.title("FITS Viewer")

# ファイル選択ボタン
open_button = tk.Button(root, text="Open FITS File", command=open_fits_file, relief="raised", width=20)
open_button.pack(pady=10)

# ヘッダー表示エリア
header_frame = tk.LabelFrame(root, text="Header", padx=10, pady=10)
header_frame.pack(padx=10, pady=10, fill="both", expand=True)
header_text = tk.Text(header_frame, wrap=tk.WORD, height=10, font=("Courier", 10), bg="#f4f4f4", fg="#000000")
header_text.pack(expand=True, fill="both")

# HDUリストエリア
hdu_frame = tk.LabelFrame(root, text="HDU List", padx=10, pady=10)
hdu_frame.pack(padx=10, pady=10, fill="both", expand=True)
hdu_listbox = tk.Listbox(hdu_frame, height=10, font=("Courier", 10), bg="#f4f4f4", fg="#000000", selectmode=tk.SINGLE)
hdu_listbox.pack(expand=True, fill="both")
hdu_listbox.bind('<<ListboxSelect>>', on_hdu_select)

# データ表示エリア
data_frame = tk.LabelFrame(root, text="Data", padx=10, pady=10)
data_frame.pack(padx=10, pady=10, fill="both", expand=True)
data_text = tk.Text(data_frame, wrap=tk.WORD, height=10, font=("Courier", 10), bg="#f4f4f4", fg="#000000")
data_text.pack(expand=True, fill="both")

# スクロールバーを追加（ヘッダー、HDUリスト、データそれぞれに追加）
header_scroll = tk.Scrollbar(header_frame, orient=tk.VERTICAL, command=header_text.yview)
header_text.config(yscrollcommand=header_scroll.set)
header_scroll.pack(side=tk.RIGHT, fill=tk.Y)

data_scroll = tk.Scrollbar(data_frame, orient=tk.VERTICAL, command=data_text.yview)
data_text.config(yscrollcommand=data_scroll.set)
data_scroll.pack(side=tk.RIGHT, fill=tk.Y)

# GUIを表示
root.mainloop()