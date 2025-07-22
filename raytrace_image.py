import matplotlib.path as mplpath
from pathlib import Path
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm   # 対数表示したい場合
import matplotlib.patches as mpatches
from regions import Regions
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from scipy.ndimage import gaussian_filter
from functools import lru_cache
from regions import Regions, RectangleSkyRegion
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
import json
from matplotlib.patches import Polygon, Rectangle
from typing import Tuple, Union, Sequence
from Resonance_Scattering_Simulation import PlotManager
P = PlotManager(spine_width=1,label_size=15)
PathLike = Union[str, Path]
"""
plot_finalpos_auto.py : finalxpos / finalypos を 2-D ヒスト画像化（自動 BINS & CUT）

使い方:
    python plot_finalpos_auto.py raytrace_image_1p8_8keV_1e5.fits
"""

"""
initialx/initialy と finalxpos/finalypos の 2-D ヒスト画像を
“横並び” (1 行 2 列) に描き、**それぞれに個別のカラーバー**を付けます。
"""

def load_data(filename):
    """FITS ファイルを読み込み、必要なデータを返す"""
    with fits.open(filename) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns"):
                cols = [c.lower() for c in hdu.columns.names]
                if {"initialx", "initialy", "finalxpos", "finalypos"} <= set(cols):
                    data = hdu.data
                    break
        else:
            raise RuntimeError("必要な列が見つかりません。")
    return data

# ----------------------------------------------------------------------
plt_pos = 100                 # 描画範囲 ±plt_pos  (両プロット共通)
bins_min, bins_max = 128, 2048   # 自動 BINS の上限・下限
# ----------------------------------------------------------------------


def make_hist(x, y):
    """CUT と BINS を自動決定し 2-D ヒストを返す"""
    p1, p99 = np.percentile(np.hstack([x, y]), [1, 99])
    CUT = np.ceil(max(abs(p1), abs(p99)))

    N = len(x)
    bins_auto = int(2 ** np.ceil(np.log2(np.sqrt(N))))
    BINS = int(np.clip(bins_auto, bins_min, bins_max))

    H, xedges, yedges = np.histogram2d(
        x, y, bins=BINS, range=[[-CUT, CUT], [-CUT, CUT]]
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return H.T, extent

def make_hist_auto_bins(x, y, bin_width=None, target_bins=300):
    """
    2-D ヒストを自動生成

    Parameters
    ----------
    x, y : array_like
        データ列（同じ単位）
    bin_width : float, optional
        1 bin の幅。単位は x, y と同じ。
        None の場合は視野サイズ / target_bins で自動決定。
    target_bins : int, optional
        bin_width を自動決定するとき「このぐらいの画素数にしたい」目安。
        （デフォルト 300 ⇒ 300×300 画像程度）
    """
    # ---- 範囲決定 ------------------------------------------------------
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])

    # “外れ値が全面を引き延ばさない” 程度に 1–99% を採用
    pad = 0.05  # 5% だけ余白を足す
    xr = (xmax - xmin)
    yr = (ymax - ymin)
    xmin -= xr * pad
    xmax += xr * pad
    ymin -= yr * pad
    ymax += yr * pad

    # ---- bin 幅 & bin 数 ------------------------------------------------
    if bin_width is None:
        # 自動: 長辺 / target_bins で幅を決め、2 のべき乗に丸める
        fov = max(xmax - xmin, ymax - ymin)
        bw  = fov / target_bins
        # きれいな数字へ (10,5,2,1,0.5,0.2,…×10^n)
        exp  = np.floor(np.log10(bw))
        base = bw / 10**exp
        nice = min([10, 5, 2, 1, 0.5, 0.2, 0.1], key=lambda n: abs(n-base))
        bin_width = nice * 10**exp
    bins_x = int(np.ceil((xmax - xmin) / bin_width))
    bins_y = int(np.ceil((ymax - ymin) / bin_width))

    H, xedges, yedges = np.histogram2d(
        x, y, bins=[bins_x, bins_y], range=[[xmin, xmax], [ymin, ymax]]
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return H.T, extent, bin_width

def plot_image(fitsfile):
    with fits.open(fitsfile) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': wcs})
    # 画像表示設定
    norm = simple_norm(data, 'log', percent=100)
    im = ax.imshow(data, origin='lower', cmap='inferno', norm=norm)
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.coords[0].set_major_formatter('d.ddd')  # RA
    ax.coords[1].set_major_formatter('d.ddd')  # Dec
    # RA/Dec ラベル表示
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    #ax.set_title("Image with DS9 Region Overlay")
    plt.tight_layout()
    return ax

def det2sky(x_arcmin, y_arcmin, ra_center_deg, dec_center_deg, roll_deg):
    """
    Convert detector–plane offsets (x,y) to sky coordinates (RA, Dec).

    Parameters
    ----------
    x_arcmin, y_arcmin : float or ndarray
        Detector offsets in arc-minutes.  x is along detector +X, y along +Y.
        Positive x points toward detector-X axis; positive y toward detector-Y axis.
    ra_center_deg, dec_center_deg : float
        Sky coordinates (deg) of the detector origin (x=y=0).
    roll_deg : float
        Rotation of the detector about the line of sight, in degrees.
        *Positive* roll is a counter-clockwise rotation of the detector axes
        with respect to the sky axes (i.e. +X rotates from +RA toward +Dec).

    Returns
    -------
    ra_deg, dec_deg : ndarray
        Sky coordinates (deg) corresponding to each (x,y) pair.
    """
    # 1) undo the roll so offsets are in sky-axis directions
    roll = np.deg2rad(roll_deg)
    x_sky =  x_arcmin * np.cos(roll) - y_arcmin * np.sin(roll)   # east (+RA)   offset [arcmin]
    y_sky =  x_arcmin * np.sin(roll) + y_arcmin * np.cos(roll)   # north (+Dec) offset [arcmin]

    # 2) convert arcmin → degrees
    dra_deg  = x_sky / 60.0 / np.cos(np.deg2rad(dec_center_deg))   # RA needs cos δ factor
    ddec_deg = y_sky / 60.0

    # 3) add to pointing
    ra_out  = (ra_center_deg + dra_deg + 360.0) % 360.0            # wrap to 0–360°
    dec_out = dec_center_deg + ddec_deg

    return ra_out, dec_out

def det2sky_offset(ra0_deg: float = 116.881311953645,
            dec0_deg: float = -19.2948862315813,
            roll_deg: float = 111.797509922333,
            x_off_arcmin: float = 1.270833,
            y_off_arcmin: float = -1.270833) -> tuple:
    """
    Parameters
    ----------
    ra0_deg, dec0_deg : float
        中心の赤経・赤緯 [deg]
    roll_deg : float
        視野の回転角 [deg]．roll=0° で検出器 +x が東 (+RA)，+y が北 (+Dec)
    x_off_arcmin, y_off_arcmin : float
        検出器座標系のオフセット [arcmin]
        (右手系: +x = detector 横方向, +y = detector 縦方向)

    Returns
    -------
    ra_deg, dec_deg : float
        オフセット後の赤経・赤緯 [deg]  (RA は 0–360° に正規化)
    """
    center = SkyCoord(ra0_deg*u.deg, dec0_deg*u.deg, frame='icrs')

    roll = np.deg2rad(-roll_deg)
    x_sky =  x_off_arcmin*np.cos(roll) - y_off_arcmin*np.sin(roll)  # [arcmin]
    y_sky =  x_off_arcmin*np.sin(roll) + y_off_arcmin*np.cos(roll)  # [arcmin]

    pa   = np.arctan2(x_sky, y_sky) * u.rad     # 0 rad = 北、π/2 rad = 東
    dist = np.hypot(x_sky, y_sky)  * u.arcmin

    target = center.directional_offset_by(pa, dist)
    return target.ra.deg, target.dec.deg

def make_ds9_box_region(filename, ra_deg, dec_deg, roll_deg, size_arcmin=3.05):
    """
    DS9の正方形box regionファイルを作成

    Parameters
    ----------
    filename : str
        出力する .reg ファイル名
    ra_deg : float
        中心RA [deg]
    dec_deg : float
        中心Dec [deg]
    roll_deg : float
        回転角 [deg]（東から北への反時計回り）
    size_arcmin : float
        一辺の長さ [arcmin]
    """
    size_arcsec = size_arcmin * 60  # DS9 box の width, height は arcsec 単位
    with open(filename, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        f.write("fk5\n")
        f.write(f"box({ra_deg:.6f},{dec_deg:.6f},{size_arcsec:.2f}\",{size_arcsec:.2f}\",{roll_deg:.6f})\n")

def make_resolve_sky_region_pixel(ra0_deg: float = 116.881311953645,
            dec0_deg: float = -19.2948862315813,
            roll_deg: float = 111.797509922333):
    """
    ra,dec,rollからresolveの各pixel resionを作成するscript
    """
    # 1 pixel = 3.05/6 arcmin
    detector_map = "/Users/keitatanaka/Dropbox/share/python_script/detector_pixel_map.json"
    # pixel mapはdetector座標系で、keynameがpixel id, それぞれ[x, y]の配列になっている
    with open(detector_map, "r") as f:
        pixel_map = json.load(f)
    pixel_center_x = 2.5
    pixel_center_y = 2.5
    pixel_size = 3.05 / 6
    # ex)[0,0]だと、3.5*pixel_size, 3.5*pixel_sizeだけ中心座標がずれる
    # それを考慮して、各pixelの中心座標を求める
    pixel_center_sky = {}
    for key, value in pixel_map.items():
        x = (value[0]-pixel_center_x) * pixel_size
        y = (value[1]-pixel_center_y) * pixel_size
        # それぞれのpixelの中心座標を求める
        ra, dec = det2sky_offset(ra0_deg, dec0_deg, roll_deg, x, y)
        # 各pixelの中心座標を保存する
        print(key, ra, dec)
        make_ds9_box_region(
            f"pixel_{key}_sky.reg", ra, dec, roll_deg, size_arcmin=pixel_size)

def combine_ds9_regions(pixelidlist, output_file):
    """
    複数の .reg ファイルを結合し、1つの .reg ファイルに保存する

    Parameters
    ----------
    filelist : list of str
        入力する .reg ファイルのリスト
    output_file : str
        出力する結合済み region ファイル
    """
    region_lines = []
    header_written = False
    filelist = [f"../regions/pixel_{name}_sky.reg" for name in pixelidlist]
    for fname in filelist:
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                # 最初に1回だけヘッダを書く
                if line.startswith("# Region file format") or line.lower() == "fk5":
                    if not header_written:
                        region_lines.append(line + "\n")
                    continue
                # その他の有効な region を追加
                if line and not line.startswith("#"):
                    region_lines.append(line + "\n")

        header_written = True

    # 出力
    with open(output_file, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("fk5\n")
        f.writelines(region_lines)

    print(f"Saved combined region to: {output_file}")

def select_region_image(region_file, image_file):
    pass

# def plot_regions(ax, regfile, **patch_kwargs):
#     """
#     DS9/CIAO リージョンを matplotlib Axes に重ね描く。

#     対応形状
#       circle / ellipse / rectangle / polygon / rectangleannulus
#     """
#     from astropy.wcs import WCS
#     from regions import RectangleSkyRegion, Regions
#     regs = Regions.read(regfile)


#     # for reg in regs:
#     #     # 除外リージョン (-box など) はスキップ
#     #     if reg.meta.get("include", True) is False:
#     #         continue

#     #     # ------------ 座標系チェック -----------------------------------
#     #     # FK5・ICRS は同じ座標値（度）で扱える
#     #     frame = getattr(reg.center.frame, "name", "").lower()
#     #     if frame not in {"fk5", "icrs", "galactic", ""}:
#     #         print(f"[plot_regions] skip unsupported frame: {frame}")
#     #         continue
#     #     if frame == "galactic":
#     #         # 必要なら ra/dec へ変換; ここでは簡略にスキップ
#     #         print("[plot_regions] galactic frame skipped")
#     #         continue

#     #     # ------------ 各形状をパッチ化 ---------------------------------
#     #     rtype = reg.__class__.__name__.lower()
#     #     print(f"[plot_regions] {rtype} {reg.center} {reg.meta}")
#     #     if rtype == "circle":
#     #         rad = reg.radius.to_value("deg")
#     #         patch = mpatches.Circle(
#     #             (reg.center.ra.deg, reg.center.dec.deg),
#     #             rad, fill=False, **patch_kwargs)

#     #     elif rtype == "ellipse":
#     #         w = reg.width.to_value("deg")
#     #         h = reg.height.to_value("deg")
#     #         patch = mpatches.Ellipse(
#     #             (reg.center.ra.deg, reg.center.dec.deg),
#     #             w, h, angle=reg.angle.to_value("deg"),
#     #             fill=False, **patch_kwargs)

#     #     elif rtype == "rectangleskyregion":
#     #         w = reg.width.to_value("deg")
#     #         h = reg.height.to_value("deg")
#     #         patch = mpatches.Rectangle(
#     #             (reg.center.ra.deg - w/2, reg.center.dec.deg - h/2),
#     #             w, h, angle=reg.angle.to_value("deg"),rotation_point='center',
#     #             fill=False, **patch_kwargs)

#     #     elif rtype == "rectangleannulusskyregion":
#     #         # 外枠を描き、中枠を破線で描く
#     #         ow = reg.outer_width.to_value("deg")
#     #         oh = reg.outer_height.to_value("deg")
#     #         iw = reg.inner_width.to_value("deg")
#     #         ih = reg.inner_height.to_value("deg")
#     #         angle = reg.angle.to_value("deg")
#     #         # outer
#     #         patch = mpatches.Rectangle(
#     #             (reg.center.ra.deg - ow/2, reg.center.dec.deg - oh/2),
#     #             ow, oh, angle=angle,
#     #             fill=False, **patch_kwargs)
#     #         ax.add_patch(patch)
#     #         # inner
#     #         inner_kw = dict(patch_kwargs)
#     #         inner_kw.setdefault("linestyle", "--")
#     #         ax.add_patch(
#     #             mpatches.Rectangle(
#     #                 (reg.center.ra.deg - iw/2, reg.center.dec.deg - ih/2),
#     #                 iw, ih, angle=angle, fill=False, **inner_kw))
#     #         print(reg.center.ra.deg, reg.center.dec.deg,
#     #               reg.outer_width, reg.outer_height,
#     #               reg.inner_width, reg.inner_height)
#     #         continue  # すでに add_patch 済みなので skip 下の routine

#     #     elif rtype == "polygon":
#     #         verts = [(v.ra.deg, v.dec.deg) for v in reg.vertices]
#     #         patch = mpatches.Polygon(verts, closed=True, fill=False, **patch_kwargs)
#     #     else:
#     #         print(f"[plot_regions] unsupported type: {rtype}")
#     #         continue

#     #     ax.add_patch(patch)
#     regs = Regions.read(regfile, format="ds9")   # ★ ここが旧 read_ds9 の代替

#     for sreg in regs:                                     # SkyRectangularRegion など
#         if isinstance(sreg, RectangleSkyRegion):
#             pix = sreg.to_pixel("fk5")                      # FK5 → pixel
#             patch = pix.as_artist(                        # as_artist() が現行名
#                 facecolor="none", edgecolor="lime", lw=1.5)
#             ax.add_patch(patch)

@lru_cache(maxsize=32)
def _read_regions(path):
    """DS9 region file → cached regions list"""
    return Regions.read(path, format="ds9")


def plot_regions(
    ax,
    regfile,
    *,
    replace=True,                   # 同じ regfile を上書き
    legend_label=None,              # 凡例に出したい名前（None → ファイル名）
    legend_loc="upper right",
    **patch_kw,
):
    """
    ● regfile 内に複数の矩形があっても、凡例は「その regfile につき 1 行」

    Parameters
    ----------
    replace : bool
        True なら同じ regfile で描いた旧パッチを掃除して描き直す。
    legend_label : str | None
        凡例に表示する文字列。None なら `Path(regfile).stem`。
    """
    reg_key = Path(regfile).resolve().as_posix()
    legend_label = legend_label or Path(regfile).stem

    # ─── 0) 既存パッチ・凡例のクリーンアップ（同じ regfile だけ） ───
    if replace:
        for p in list(ax.patches):
            if getattr(p, "_reg_key", None) == reg_key:
                p.remove()

        # 既存凡例から同じキーの行だけ除去
        leg = ax.get_legend()
        if leg:
            h_old, l_old = ax.get_legend_handles_labels()
            keep = [(h, l) for h, l in zip(h_old, l_old)
                    if getattr(h, "_reg_key", None) != reg_key]
            leg.remove()                # 全消し
            if keep:                    # 残すものがあれば再構築
                ax.legend(*zip(*keep), loc=legend_loc,
                          fontsize="small", framealpha=0.8, edgecolor="k")

    # ─── 1) 新しいパッチを描画 ───
    regs = _read_regions(regfile)
    wcs  = ax.wcs

    rep_handle = None                  # 凡例用の代表パッチ
    for sreg in regs:
        if not isinstance(sreg, RectangleSkyRegion):
            continue

        pix = sreg.to_pixel(wcs)
        patch = pix.as_artist(
            facecolor="none",
            label=legend_label,        # ← 全パッチ同じ label
            **patch_kw
        )
        patch._reg_key = reg_key       # 自前タグ
        ax.add_patch(patch)

        # 1 個目を凡例の代表ハンドルにする
        if rep_handle is None:
            rep_handle = patch

    # ─── 2) 凡例を 1 行だけ追加 ───
    if rep_handle is not None:
        h_cur, l_cur = ax.get_legend_handles_labels()
        if legend_label not in l_cur:              # 二重追加を防ぐ
            h_cur.append(rep_handle)
            l_cur.append(legend_label)
            ax.legend(h_cur, l_cur, loc=legend_loc,
                      fontsize="small", framealpha=0.8, edgecolor="k")


"""
Rotate the *grid*, not the data.

* Detector coordinates : x,y = −0.5 … 5.5  (6×6 pix)
* Rotation centre      : (2.5, 2.5) pix
* No resampling / interpolation – the raw counts stay put.
"""


def plot_image_and_region_pks(fitsfile):
    ax = plot_image(fitsfile)
    plot_regions(ax, '../../outer_sky.reg', edgecolor="cyan", linewidth=1.2)
    plot_regions(ax, '../../center_sky.reg', edgecolor="green", linewidth=1.2)
    plot_regions(ax, '../../exMXS_sky.reg', edgecolor="blue", linewidth=1.2)
    # --- ズーム範囲を RA/Dec (deg) → pixel に変換して設定 ------------------
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # 希望する RA/Dec 範囲
    ra_min, ra_max = 116.88 - 0.05, 116.88 + 0.05
    dec_min, dec_max = -19.30 - 0.05, -19.30 + 0.05

    # WCS 取得
    wcs = ax.wcs

    # 4 隅を pixel に変換
    x0, y0 = wcs.world_to_pixel(SkyCoord(ra_min * u.deg, dec_min * u.deg))
    x1, y1 = wcs.world_to_pixel(SkyCoord(ra_max * u.deg, dec_max * u.deg))

    # set_xlim / set_ylim は pixel 座標で指定
    ax.set_xlim(min(x0, x1), max(x0, x1))
    ax.set_ylim(min(y0, y1), max(y0, y1))
    counts_center = count_counts_in_regions(fitsfile, '../../center_sky.reg')
    counts_outer = count_counts_in_regions(fitsfile, '../../outer_sky.reg')
    count_exMXS = count_counts_in_regions(fitsfile, '../../exMXS_sky.reg')
    print(counts_center, np.sum(counts_outer), np.sum(count_exMXS))
    plt.savefig('pks_from_center_counts.pdf',dpi=300,transparent=False)
    plt.show()

def main(fname="raytrace_image_SSM_outer_to_center_1p8_8keV_1e5.fits"):
    data = load_data(fname)
# ヒスト生成 -------------------------------------------------------------
    hist_init, ext_init = make_hist(data["initialx"],  data["initialy"])
    hist_final, ext_final = make_hist(data["finalxpos"], data["finalypos"])
    hist_initang, ext_initang = make_hist(data["initialtheta"], data["initialazimdir"]/60)
    # 描画 -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=False, sharey=False)

    for ax, H, ext, xlabel, ylabel in [
            (axes[0], hist_init,  ext_init,  "initialx", "initialy"),
            (axes[1], hist_final, ext_final, "finalxpos", "finalypos"),
            (axes[2], hist_initang, ext_initang, "initialtheta", "initialazimdir")
            ]:

        im = ax.imshow(
            H, origin="lower", extent=ext, cmap="inferno",
            norm=LogNorm(vmin=1, vmax=H.max())
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-plt_pos, plt_pos)
        ax.set_ylim(-plt_pos, plt_pos)
        ax.set_title(f"{xlabel.split('x')[0].capitalize()} distribution")

        # ─ 個別カラーバー ─
        fig.colorbar(im, ax=ax, label="Counts / pixel")

        fig.suptitle(fname)
        fig.tight_layout()
    plt.show()

def main_radec(fname="raytrace_image_SSM_exMXS_to_center_1p8_8keV_1e6.fits",
               bin_width_arcmin=None, region_file="./regions/all_sky.reg"):
    """
    detector → sky 変換後に 2-D ヒストを横並び表示

    Parameters
    ----------
    fname : str
        FITS ファイル名
    bin_width_arcmin : float or None
        1 bin 幅（arcmin）。None なら自動で良い感じに決める
    """
    data = load_data(fname)
    ra_cen = 116.881311953645
    dec_cen = -19.2948862315813
    roll = -111.797509922333
    mm2arcmin = 6.138833e-1
    optical_axis_offset_x = -2.0/60 # arcmin
    optical_axis_offset_y = 15/60 # arcmin
    # optical_axis_offset_x = 0 # arcmin
    # optical_axis_offset_y = 0 # arcmin

    init_ra, init_dec = det2sky(data["initialx"]*mm2arcmin+optical_axis_offset_x,
                                data["initialy"]*mm2arcmin+optical_axis_offset_y,
                                ra_cen, dec_cen, roll)
    final_ra, final_dec = det2sky(data["finalxpos"]*mm2arcmin+optical_axis_offset_x,
                                  data["finalypos"]*mm2arcmin+optical_axis_offset_y,
                                  ra_cen, dec_cen, roll)

    # --- ヒスト生成（RA/Dec は度）--------------------------------------
    # bin 幅指定が arcmin → 度に直す
    bw_deg = None if bin_width_arcmin is None else bin_width_arcmin/60.0
    hist_init, ext_init, bw = make_hist_auto_bins(init_ra, init_dec, bw_deg)
    hist_final, ext_final, _ = make_hist_auto_bins(final_ra, final_dec, bw_deg)

    print(f"auto bin width  ≈ {bw*60:.3f} arcmin  ({bw:.5f} deg)")

    # --- 描画 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)

    for ax, H, ext, title in [
            (axes[0], hist_init,  ext_init,  "Initial (RA,Dec)"),
            (axes[1], hist_final, ext_final, "Final (RA,Dec)")]:
        im = ax.imshow(H, origin="lower", extent=ext, cmap="inferno",
                       norm=LogNorm(vmin=1, vmax=H.max()))
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Counts / pixel")

        # ─ リージョンの重ね描き ─
        if region_file:
            plot_regions(ax, region_file,
                         edgecolor="cyan", linewidth=1.2)

    fig.suptitle(Path(fname).name)
    fig.tight_layout()
    plt.show()
    fig.savefig(Path(fname).with_suffix(".png"), dpi=300)

def count_counts_in_regions(image_fits: str,
                            region_file: str,
                            mask_mode: str = "center") -> list:
    """
    Sum counts inside each DS9 region over the given FITS image.

    Parameters
    ----------
    image_fits : str
        2-D FITS 画像へのパス（pixel がカウント値）。
    region_file : str
        DS9 region ファイル（fk5 想定）。
    mask_mode : {"center", "exact"}, optional
        regions.Region.to_mask() に渡すモード。
        "center": 高速（pixel 中心で判定）／"exact": 端を考慮。

    Returns
    -------
    list[float]
        .reg に並んでいる順でのカウント合計。
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from regions import Regions

    # 画像＆WCS 読み込み
    with fits.open(image_fits) as hdul:
        data   = hdul[0].data
        header = hdul[0].header
        wcs    = WCS(header)

    regs = Regions.read(region_file, format="ds9")

    counts = []
    for reg in regs:
        # Sky → pixel
        pix_reg = reg.to_pixel(wcs)

        # マスク生成
        mask = pix_reg.to_mask(mode=mask_mode)

        # マスク適用
        cutout = mask.cutout(data)
        if cutout is None:           # 画面外など
            counts.append(0.0)
            continue

        counts.append(float(np.sum(cutout * mask.data)))

    return counts

def plot_image_and_region(fitsfile, regfile):
    ax = plot_image(fitsfile)
    plot_regions(ax, regfile, edgecolor="cyan", linewidth=1.2)
    # --- ズーム範囲を RA/Dec (deg) → pixel に変換して設定 ------------------
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # 希望する RA/Dec 範囲
    ra_min, ra_max = 116.88 - 0.05, 116.88 + 0.05
    dec_min, dec_max = -19.30 - 0.05, -19.30 + 0.05

    # WCS 取得
    wcs = ax.wcs

    # 4 隅を pixel に変換
    x0, y0 = wcs.world_to_pixel(SkyCoord(ra_min * u.deg, dec_min * u.deg))
    x1, y1 = wcs.world_to_pixel(SkyCoord(ra_max * u.deg, dec_max * u.deg))

    # set_xlim / set_ylim は pixel 座標で指定
    ax.set_xlim(min(x0, x1), max(x0, x1))
    ax.set_ylim(min(y0, y1), max(y0, y1))
    counts = count_counts_in_regions(fitsfile, regfile)
    print(counts)
    plt.show()

def _coord_to_index(lo: float, hi: float) -> tuple[int, int]:
    return int(np.floor(lo + 0.5)), int(np.floor(hi - 0.5))   # −0.5 →0, 5.5→5

def plot_image_and_region_wsmooth(
    fitsfile,
    regfile,
    *,
    sigma_pix=3,           # ガウシアン平滑の σ [pixel]
    contour_levels=None,     # 省略時は自動設定 (パーセンタイル)
    cmap="inferno",
):
    """
    FITS 画像に
      1. 全体をガウシアン平滑
      2. 輪郭線 (contour)
      3. DS9 region の描画
    をまとめて行う。

    Parameters
    ----------
    sigma_pix : float
        Gaussian smoothing の σ [pixel]．
    contour_levels : sequence[float] | None
        輪郭線の強度。データ値そのものでもパーセンタイルでも可。
        None のときは 60–97 %tile を 5 本自動生成。
    """

    # ---------- 画像を読み込み，平滑化 ----------
    with fits.open(fitsfile) as hdul:
        hdu   = hdul[0]
        data  = hdu.data.astype(float)            # 必ず float に
        wcs   = WCS(hdu.header)

    if sigma_pix != 0:
        smooth = ds9_gaussian_smooth(data, radius_pix=sigma_pix)
    else:
        smooth = data
        print("non smooth")

    # ---------- figure / axes 生成 ----------
    fig   = plt.figure(figsize=(6, 6))
    ax    = fig.add_subplot(
        111, projection=wcs, facecolor="k")       # projection=WCS で天球座標
    positive = smooth[smooth > 0]           # log なので >0 だけ抜き出す
    if positive.size == 0:
        raise ValueError("画像に正の値がありません。log 表示できません。")
    vmin, vmax = np.nanpercentile(positive, (0, 99.5))

    im = ax.imshow(
        smooth,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax), # ★ ここを LogNorm に
    )

    # ---------- 輪郭線 ----------
    if contour_levels is None:
        contour_levels = np.nanpercentile(smooth, [60, 70, 80, 90, 97])
    # ax.contour(
    #     smooth,
    #     levels=contour_levels,
    #     colors="white",
    #     linewidths=0.8,
    #     transform=ax.get_transform("pixel"),      # ★ pixel → sky 変換
    # )

    # ---------- DS9 region ----------
    plot_regions(ax, 'exMXS_sky.reg', edgecolor="black", linewidth=1.2,linestyle="dotted",legend_label="ex12pix")
    plot_regions(ax, 'outer_MXS_sky.reg', edgecolor="blue", linewidth=1.2,linestyle="dashed",legend_label="outer")
    plot_regions(ax, 'center_sky_sky.reg', edgecolor="red", linewidth=1.2,linestyle="-",legend_label="center")

    # ---------- ズーム範囲を設定 ----------
    ra_min, ra_max = 116.881311953645 - 0.05, 116.881311953645 + 0.05
    dec_min, dec_max = -19.2948862315813 - 0.05, -19.2948862315813 + 0.05
    x0, y0 = wcs.world_to_pixel(SkyCoord(ra_min * u.deg, dec_min * u.deg))
    x1, y1 = wcs.world_to_pixel(SkyCoord(ra_max * u.deg, dec_max * u.deg))
    ax.set_xlim(min(x0, x1), max(x0, x1))
    ax.set_ylim(min(y0, y1), max(y0, y1))

    # ---------- ラベルなど ----------
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Counts (smoothed)")

    # ---------- 集計 ----------
    counts = count_counts_in_regions(fitsfile, regfile)
    print("Counts in each region:", counts)
    plt.tight_layout()
    plt.show()

def plot_grid_rotated(
    fits_path: str | Path,
    roll_deg: float,
    *,
    xy_range: tuple[float, float] = (-0.5, 5.5),
    cmap: str = "gray",
    show_pixel_grid: bool = True,
    ax: plt.Axes | None = None,
):
    with fits.open(fits_path) as hdul:
        data_full = hdul[0].data.astype(float)

    # 6×6 部分を切り出し
    ix0 = int(np.floor(xy_range[0] + 0.5))
    ix1 = int(np.floor(xy_range[1] - 0.5))
    iy0 = ix0
    iy1 = ix1
    patch = data_full[iy0:iy1 + 1, ix0:ix1 + 1]

    if ax is None:
        ax = plt.gca()

    # 画像を回さず「枠」だけ回転
    trans = Affine2D().rotate_deg_around(2.5, 2.5, roll_deg)
    im = ax.imshow(
        patch,
        origin="lower",
        cmap=cmap,
        interpolation="none",
        extent=[*xy_range, *xy_range],
        transform=trans + ax.transData,
    )

    # ピクセル線
    if show_pixel_grid:
        for i in range(7):
            ax.plot([-0.5, 5.5], [i - .5, i - .5],
                    lw=.3, color="w", alpha=.6,
                    transform=trans + ax.transData)
            ax.plot([i - .5, i - .5], [-0.5, 5.5],
                    lw=.3, color="w", alpha=.6,
                    transform=trans + ax.transData)
    # ax.set_xlim(xy_range)
    # ax.set_ylim(xy_range)
    ax.set_aspect("equal")
    # ax.set_xlabel("DETX [pixel]")
    # ax.set_ylabel("DETY [pixel]")

    return im                     # ← 呼び出し側でカラーバーを付ける

def make_resolve_wcs_noroll(ra_centre_deg: float,
                            dec_centre_deg: float,
                            pix_scale_arcmin: float = 3.05/6) -> WCS:
    """6×6 pix Resolve WCS (roll=0°, CRPIX=3.5)."""
    scale_deg = pix_scale_arcmin / 60.0
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_centre_deg, dec_centre_deg]
    w.wcs.crpix = [3.5, 3.5]                      # ← 中心 (2.5,2.5)
    w.wcs.cd    = [[-scale_deg, 0.0],
                   [ 0.0,       scale_deg]]
    return w


def imshow_resolve_rotated(
    fits_path: PathLike,
    roll_deg: float,
    *,
    ax: plt.Axes,
    wcs_no_roll: WCS,
    cmap="plasma",
    inner_color="white",
    outer_color="white",
    lw=3,
    shrink_margin=0.05,        # ← 何 pix 内側に描くか
):
    # --- 6×6 データ ---
    patch = fits.getdata(fits_path, 0)[:6, :6].astype(float)

    # --- 回転変換 ---
    rot = Affine2D().rotate_deg_around(2.5, 2.5, roll_deg)
    tr  = rot + ax.get_transform("pixel")

    im = ax.imshow(
        patch, origin="lower", cmap=cmap, interpolation="none",
        extent=[-0.5, 5.5, -0.5, 5.5], transform=tr
    )

    # ===== inner (中央 2×2) =====
    # ax.add_patch(Rectangle(
    #     (1.5, 1.5), 2, 2,
    #     fill=False, edgecolor=inner_color, lw=lw, ls="-", transform=tr,
    #     label="center"
    # ))
    # ひと回り小さい枠（中心縮小）
    m = shrink_margin
    ax.add_patch(Rectangle(
        (1.5 + m, 1.5 + m), 2 - 2*m, 2 - 2*m,
        fill=False, edgecolor=inner_color, lw=lw, ls="--", transform=tr,
        label="center"
    ))

    # ===== outer =====
    verts_outer  = [
        (-0.5, 5.5), (3.5, 5.5), (3.5, 3.5), (1.5, 3.5),
        (1.5, 1.5), (3.5, 1.5), (3.5, -0.5), (0.5, -0.5),
        (0.5, 0.5), (-0.5, 0.5), (-0.5, 5.5)
    ]

    # “縮小 outer” は 新関数で作る
    verts_shrink = make_outer_shrink_vertices(m=shrink_margin)

    ax.add_patch(Polygon(
        verts_outer,  closed=True, fill=False,
        edgecolor=outer_color, lw=lw, transform=tr, label="outer"
    ))
    # ax.add_patch(Polygon(
    #     verts_shrink, closed=True, fill=False,
    #     edgecolor=outer_color, lw=lw, ls="--", transform=tr,
    #     label="outer shrink"
    # ))

    # --- 軸体裁 & 凡例 ---
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.coords[0].set_axislabel("RA (deg)")
    ax.coords[1].set_axislabel("Dec (deg)")
    leg = ax.legend(framealpha=0.9,
                    facecolor="#888888",    # 暗背景
                    edgecolor="white",
                    labelcolor="white",
                    loc = "upper left")     # Matplotlib ≥3.7
    return im
# ---- 6×6 patch を返すだけのヘルパ -----------------------------
def get_resolve_patch(fits_path: str | Path,
                      xy_range=(-0.5, 5.5)) -> np.ndarray:
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
    i0 = int(np.floor(xy_range[0] + 0.5))
    i1 = int(np.floor(xy_range[1] - 0.5))
    return data[i0:i1+1, i0:i1+1]            # (6, 6)
# ---------------------------------------------------------------
# ----------------------------------------------------------------------
# 2) Chandra + RESOLVE の 2 段表示
# ----------------------------------------------------------------------

def chandra_with_resolve_image(
    chandra_image: str,
    resolve_image: str,
    *,
    ra_centre: float = 116.881311953645,          # [deg]
    dec_centre: float = -19.2948862315813,         # [deg]
    roll: float = 111.797509922333,       # [deg]
    sigma_pix: float = 0,
    cmap_chandra: str = "ds9_b",
    cmap_resolve: str = "ds9_b",
    layout: str = "horizontal",          # 横並び
):
    import ds9_colors
    # ----- Chandra -----
    with fits.open(chandra_image) as hdul:
        ch_data = hdul[0].data.astype(float)
        ch_wcs  = WCS(hdul[0].header)

    ch_smooth = gaussian_filter(ch_data, sigma=sigma_pix) if sigma_pix else ch_data
    # ch_smooth = ch_data
    vmin, vmax = np.nanpercentile(ch_smooth[ch_smooth > 0], (0, 99.5))

    # ----- Figure layout -----
    fig = plt.figure(figsize=(12, 6))
    gs  = fig.add_gridspec(1, 2, wspace=0.001)

    ax1 = fig.add_subplot(gs[0], projection=ch_wcs, facecolor="black")
    ax1.set_facecolor("black")
    # Resolve 用 WCS (roll = 0°)
    res_wcs0 = make_resolve_wcs_noroll(ra_centre, dec_centre)
    ax2 = fig.add_subplot(gs[1], projection=res_wcs0, facecolor="black")
    ax2.set_facecolor("black")
    fig.subplots_adjust(
        top    = 0.85,   # ↑色バーが収まる高さまで下げる
        right  = 0.98,
        left   = 0.02,
        wspace = 0.0,
        bottom = 0.02,   # 下余白（必要なら調整）
    )
    # ----- plot Chandra -----
    im1 = ax1.imshow(ch_smooth, origin="lower",
                     cmap=cmap_chandra, norm=LogNorm(vmin=vmin, vmax=vmax))
    ax1.coords[0].set_axislabel("RA")
    ax1.coords[1].set_axislabel("Dec")
    for ax in (ax1,):
        lon, lat = ax.coords
        lon.set_format_unit(u.deg)
        lat.set_format_unit(u.deg)
        lon.set_major_formatter('d.dd')      # RA も 10進度
        lat.set_major_formatter('d.dd')
        lon.set_axislabel("RA (deg)")
        lat.set_axislabel("Dec (deg)")
    cb1_ax = ax1.inset_axes([0.0, 1.02, 1.0, 0.05])
    cb1 = fig.colorbar(im1, cax=cb1_ax, orientation="horizontal")
    cb1.set_label("Counts/s")
    cb1.ax.xaxis.set_ticks_position("top")
    cb1.ax.xaxis.set_label_position("top")
    cb1.ax.invert_xaxis()
    dec_max=dec_centre + 0.04
    dec_min=dec_centre - 0.04
    ra_c             = ra_centre             # 中心 RA で OK
    ra_max = ra_centre + 0.04
    ra_min = ra_centre - 0.04
    # ── world (deg) → pixel へ変換 ────────────────
    x0, y0 = ax1.wcs.world_to_pixel(SkyCoord(ra_max*u.deg, dec_min*u.deg))
    x1, y1 = ax1.wcs.world_to_pixel(SkyCoord(ra_min*u.deg, dec_max*u.deg))

    ax1.set_ylim(y0, y1)  
    ax1.set_xlim(x0, x1)  
    plot_regions(ax1, 'all_sky.reg', edgecolor="white", linewidth=3,linestyle="-",legend_label="center")
    # ----- plot Resolve (rotated) -----
    im2 = imshow_resolve_rotated(
        resolve_image,
        roll,                     # 画像を回す角度
        ax=ax2,
        wcs_no_roll=res_wcs0,     # roll を含まない WCS を渡す
        cmap=cmap_resolve,
        inner_color="black",
        outer_color="white",
    )
    for ax in (ax2,):
        lon, lat = ax.coords
        lon.set_format_unit(u.deg)
        lat.set_format_unit(u.deg)
        lon.set_major_formatter('d.dd')      # RA も 10進度
        lat.set_major_formatter('d.dd')
        # lon.set_axislabel("RA (deg)")
        # lat.set_axislabel("Dec (deg)")
        lon.set_ticklabel_visible(False)   # RA の数字オフ
        lat.set_ticklabel_visible(False)   # Dec の数字オフ
    cb2_ax = ax2.inset_axes([0.0, 1.02, 1.0, 0.05])
    cb2 = fig.colorbar(im2, cax=cb2_ax, orientation="horizontal")
    cb2.set_label("Counts")
    cb2.ax.xaxis.set_ticks_position("top")
    cb2.ax.xaxis.set_label_position("top")
    cb2.ax.invert_xaxis()
    x0, y0 = ax2.wcs.world_to_pixel(SkyCoord(ra_max*u.deg, dec_min*u.deg))
    x1, y1 = ax2.wcs.world_to_pixel(SkyCoord(ra_min*u.deg, dec_max*u.deg))

    ax2.set_ylim(y0, y1)  
    ax2.set_xlim(x0, x1)  

    #plot_regions(ax2, 'center_sky_sky.reg', edgecolor="white", linewidth=1.2,linestyle="-",legend_label="center")
    #plot_regions(ax2, 'outer_MXS_sky.reg', edgecolor="white", linewidth=1.2,linestyle="-",legend_label="center")
    plt.show()
    fig.savefig("chandra_with_resolve_image.pdf")

def make_outer_shrink_vertices(m: float = 0.2):
    """
    Return vertices for the shrunken-outer polygon.
    * m : margin [pixel] to step INWARD **only** along the detector外周.
      → 内側に寄せても inner 2×2 に重ならない座標を返す
    """
    # shorthand
    xL  = -0.5 + m          # 左端
    xL2 =  0.5 + m          # 左下 notch 右端
    xC  =  1.5              # 中央ブロック左端 (動かさない)
    xR  =  3.5 - m          # 右端 (縮小)

    yT  =  5.5 - m          # 上端
    yC1 =  3.5              # 中央ブロック上端 (固定)
    yC2 =  1.5              # 中央ブロック下端 (固定)
    yL  =  0.5 - m          # notch 上端
    yB  = -0.5 + m          # 下端

    return [
        (xL, yT), (xR, yT),           # 上辺
        (xR, yC1), (xC, yC1),         # 中央上へ折れ込む
        (xC, yC2), (xR, yC2),         # 中央右側面
        (xR, yB), (xL2, yB),          # 下辺
        (xL2, yL), (xL, yL),          # notch を迂回
        (xL, yT)                      # 戻って閉じる
    ]


def ds9_gaussian_smooth(data, radius_pix, sigma_pix=None):
    """
    DS9 の Smoothing (Gaussian) を忠実に模倣するフィルタ.
    radius_pix : DS9 ダイアログで指定する 'Radius' [pixel]
    sigma_pix  : DS9 で 'Sigma' を明示的に設定した場合のみ指定
                None のときは σ = radius/2 を採用（DS9 初期値）
    """
    if sigma_pix is None:
        sigma_pix = radius_pix / 2.0          # DS9 の既定
    truncate = radius_pix / sigma_pix        # => カーネル幅 = 2*radius+1
    return gaussian_filter(data,
                        sigma=sigma_pix,
                        truncate=truncate,
                        mode="nearest")    # DS9 と同じ端処理