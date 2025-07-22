from PIL import Image, ImageDraw, ImageFont
import glob
import os

def create_pdf(input_folder, output_pdf):
    # すべてのPNGファイルを取得
    png_files = sorted(list(glob.glob(f"{input_folder}/*.png")))

    # PDFを作成するためのイメージリストを初期化
    images = []

    # 2行2列のイメージを保持する2次元リストを初期化
    image_grid = [[], []]

    # 2行2列のイメージを保持するためのカウンタ
    row_count = 0
    col_count = 0

    # PNGファイルを処理
    for png_file in png_files:
        # PILでイメージを開く
        img = Image.open(png_file)

        # イメージにファイル名を描画する
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # デフォルトのフォントを使用
        draw.text((10, 10), os.path.basename(png_file), fill="red", font=font, fontsize=40)  # ファイル名を描画

        # イメージを2次元リストに追加
        image_grid[row_count].append(img)

        # 列カウンタをインクリメント
        col_count += 1

        # 列が2に達した場合は行カウンタをインクリメントして列カウンタをリセット
        if col_count == 2:
            row_count += 1
            col_count = 0

        # 行が2に達した場合は、2行2列のイメージを作成してリセット
        if row_count == 2:
            # 2行2列のイメージを作成
            new_img = Image.new('RGB', (image_grid[0][0].width * 2, image_grid[0][0].height * 2))

            # イメージをペースト
            for i in range(2):
                for j in range(2):
                    new_img.paste(image_grid[i][j], (image_grid[0][0].width * j, image_grid[0][0].height * i))

            # PDFに変換するためのイメージをリストに追加
            images.append(new_img)

            # 2次元リストをリセット
            image_grid = [[], []]
            row_count = 0

    # 2行2列に並べられない画像が残っている場合
    if row_count > 0:
        # 不足しているイメージを空のイメージで埋める
        for i in range(row_count, 2):
            image_grid[i].append(Image.new('RGB', (image_grid[0][0].width, image_grid[0][0].height)))

        # 2行2列のイメージを作成
        new_img = Image.new('RGB', (image_grid[0][0].width * 2, image_grid[0][0].height * 2))

        # イメージをペースト
        for i in range(2):
            for j in range(2):
                new_img.paste(image_grid[i][j], (image_grid[0][0].width * j, image_grid[0][0].height * i))

        # PDFに変換するためのイメージをリストに追加
        images.append(new_img)

    # PDFに変換
    images[0].save(output_pdf, save_all=True, append_images=images[1:], resolution=100.0)

