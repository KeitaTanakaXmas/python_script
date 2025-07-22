import os

def process_files(file_extension):
    """
    Process files with the specified file extension using DS9.

    Args:
        file_extension (str): The file extension of the files to be processed.

    Returns:
        None
    """
    file_list = [filename for filename in os.listdir() if filename.endswith(file_extension)]
    for filename in file_list:
        filename_no_ext = os.path.splitext(filename)[0]
        os.system(f'ds9 -scale mode 99.5 -scale log -bin factor 4 -cmap b -rotate 2049 "{filename}" -saveimage "../fig/{filename_no_ext}_removed_img.png" -exit')

if __name__ == "__main__":
    process_files(".evt")
    process_files(".fpix")
