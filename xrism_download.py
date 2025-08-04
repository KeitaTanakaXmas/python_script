import subprocess
from pathlib import Path

import gnupg, glob, os, getpass
import subprocess, shlex, getpass
def xrism_data_download(obsid: str, outdir: Path | str = "./") -> None:
    """
    Download XRISM rev3 data **only for the given OBSID**.

    Parameters
    ----------
    obsid : str
        e.g. "000112000"
    outdir : str or pathlib.Path, optional
        download destination (default: current dir)
    """
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    first_char = obsid[0]
    base = f"/pub/xrism/data/obs/rev3/{first_char}/{obsid}"

    cmd = [
        "wget",
        "-nv",            # terse log
        "-r",             # recursive (instead of -m)
        "-np",            # no parent
        "-nH",            # ignore host dir
        "--cut-dirs=6",   # strip leading path segments
        "-R", "index.html*",           # skip index pages
        "-I", base,                     
        "--execute", "robots=off",
        "--wait=1",
        "-P", str(outdir),
        f"https://data.darts.isas.jaxa.jp{base}/",
    ]

    subprocess.run(cmd, check=True)
    print("✅ Finished :", obsid, "→", outdir)

# 例
# xrism_data_download("000112000", "~/xrism_downloads")


def multi(obsids):
    for obsid in obsids:
        xrism_data_download(obsid)


def decrypt_gpg(obsid, decrypt_key):
    gpg       = gnupg.GPG()                 # gpg コマンドが PATH にあれば OK
    passphrase = getpass.getpass()

    for path in glob.glob(f'./{obsid}/**/*.gpg', recursive=True):
        out_path = path[:-4]                # .gpg → 元ファイル名
        with open(path, 'rb') as f:
            dec_status = gpg.decrypt_file(f,
                                        passphrase=decrypt_key,
                                        output=out_path)
        if not dec_status.ok:
            print(f"fail: {path} – {dec_status.status}")

# def decrypt_gpg(obsid: str, decrypt_key: str, root='./'):
#     gpg = gnupg.GPG(                     # ★ 追加オプション
#         options=['--batch', '--yes', '--pinentry-mode', 'loopback']
#     )
#     gpg.encoding = 'utf-8'               # 日本語コメントがある場合は念のため

#     for path in pathlib.Path(root, obsid).rglob('*.gpg'):
#         out_path = path.with_suffix('')  # .gpg → 元ファイル名
#         with path.open('rb') as f:
#             status = gpg.decrypt_file(
#                 f,
#                 passphrase=decrypt_key,  # ここで渡したものがそのまま使われる
#                 output=str(out_path)
#             )
#         if not status.ok:
#             print(f'FAIL {path}: {status.status}')