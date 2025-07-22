import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
import os
from pathlib import Path
from typing import Any, Dict
from Resonance_Scattering_Simulation import Simulation

#os.chdir("/Users/keitatanaka/Dropbox/AO2_proposal/cluster_catalog/chandra_accept")
S = Simulation(cluster_name="dummy")
file_path = "/Users/keitatanaka/Dropbox/AO2_proposal/cluster_catalog/chandra_accept/all_profiles.dat"

def get_average_abundance_by_name(name: str, path: str = "average_abundances.csv") -> float | None:
    """
    平均abundanceのCSVを読み込み、指定したnameに対応するFe値を返す。

    Parameters:
    - name: str - クラスター名
    - path: str - CSVファイルのパス（デフォルト: average_abundances.csv）

    Returns:
    - float: Fe値（見つからなければ None）
    """
    try:
        df = pd.read_csv(path)
        row = df[df["name"] == name]
        if not row.empty:
            return float(row["Abund"].values[0])
        else:
            print(f"⚠️ name '{name}' not found in {path}")
            return 0.4
    except FileNotFoundError:
        print(f"❌ ファイルが見つかりません: {path}")
        return None

def read_profile(file_path=file_path):
    # ファイルを読み込む
    data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None)

    # 必要なカラムを抽出する
    # 天体名、Rin, Rout, nelec, Kitpl のカラムを選択 (データ構造に基づく)
    data.columns = ['name', 'Rin', 'Rout', 'nelec', 'neerr', 'Kitpl', 'Kflat', 'Kerr', 'Pitpl', 'Pflat', 'Perr', 
                    'Mgrav', 'Merr', 'Tx', 'Txerr', 'Lambda', 'tcool5_2', 't52err', 'tcool3_2', 't32err']

    # 各天体ごとにデータを抽出して処理
    unique_names = data['name'].unique()
    interpolated_ne = []
    interpolated_kt = []
    interpolated_Z = []
    ne_data = []
    kt_data = []
    rad_data = []
    radii = np.linspace(0, 100, 1000)  # 内挿する半径範囲 (例: Rin ~ Rout)
    print(unique_names)
    print(len(unique_names))
    for name in unique_names:
        # 名前ごとにデータをフィルタリング
        group = data[data['name'] == name]
        
        # Rin, Rout, nelec, Kitpl のデータを取り出す
        Rin = group['Rin'].values*1e+3
        Rout = group['Rout'].values*1e+3
        Rcen = (Rin + Rout) / 2
        nelec = group['nelec'].values
        Tx = group['Tx'].values
        
        # R の範囲に基づいて一次近似を行い、内挿
        ne_interp = interp1d(Rcen,nelec, kind='linear', bounds_error=False, fill_value='extrapolate')
        kt_interp = interp1d(Rcen,Tx, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # 内挿結果を保存
        interpolated_ne.append(ne_interp)
        interpolated_kt.append(kt_interp)
        ne_data.append(nelec)
        kt_data.append(Tx)
        rad_data.append(Rcen)
        z = get_average_abundance_by_name(name)
        Z = np.full_like(radii, 0.4)
        f_Z = interp1d(radii, Z, kind='linear', fill_value='extrapolate')
        interpolated_Z.append(f_Z)
    return unique_names, interpolated_ne, interpolated_kt, ne_data, kt_data, rad_data, interpolated_Z

def read_profile_unique(
    file_path: str | Path,
    target_names: str | list[str] | None = None,
    r_interp: np.ndarray | None = np.linspace(0, 1000, 1000)  # kpc
) -> Dict[str, Dict[str, Any]]:
    """
    ACCEPT 形式の all_profiles.dat から
    指定クラスターの ne(r), kT(r) プロファイルを読み込む。

    Returns
    -------
    { name : {
        'r'      : Rcen (unsorted),
        'ne'     : ne   (unsorted),
        'kt'     : Tx   (unsorted),
        'f_ne'   : 1D 補間関数,
        'f_kt'   : 1D 補間関数,
        'ne_i'   : ne(r_interp)  ※ r_interp が None でなければ,
        'kt_i'   : kT(r_interp),
        'f_Z'    : 金属量補間関数 (定数 0.4 Solar)
      } }
    """
    # 1) ファイル読み込み
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment="#",
        header=None,
        skip_blank_lines=True,
    )

    base_cols = [
        "name", "Rin", "Rout", "nelec", "neerr",
        "Kitpl", "Kflat", "Kerr",
        "Pitpl", "Pflat", "Perr",
        "Mgrav", "Merr",
        "Tx", "Txerr",
        "Lambda", "tcool5_2", "t52err", "tcool3_2", "t32err",
    ]
    ncols = df.shape[1]
    df.columns = base_cols[:ncols] + [f"col{i}" for i in range(ncols - len(base_cols))]

    # 数値化
    num_cols = [c for c in df.columns if c != "name"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 名前フィルタ
    if target_names is not None:
        if isinstance(target_names, str):
            target_names = [target_names]
        target_set = {n.strip().upper() for n in target_names}
        df = df[df["name"].str.strip().str.upper().isin(target_set)]

    profiles = {}
    for name, grp in df.groupby("name", sort=False):
        # ---- 半径・物理量を取得 ----
        Rin  = grp["Rin"].values   * 1e3       # すでに kpc 単位
        Rout = grp["Rout"].values  * 1e3
        Rcen = 0.5 * (Rin + Rout)

        ne   = grp["nelec"].values
        Tx   = grp["Tx"].values

        # ---- Rcen 昇順にソート ----
        idx  = np.argsort(Rcen)
        Rcen = Rcen[idx]
        ne   = ne[idx]
        Tx   = Tx[idx]

        # ---- 補間関数 ----
        f_ne = interp1d(Rcen, ne, kind="linear",
                        bounds_error=False, fill_value="extrapolate")
        f_kt = interp1d(Rcen, Tx, kind="linear",
                        bounds_error=False, fill_value="extrapolate")

        # ---- r_interp グリッドに補間した配列も返す ----
        if r_interp is not None:
            ne_i = f_ne(r_interp)
            kt_i = f_kt(r_interp)
        else:
            ne_i = kt_i = None

        # ---- 金属量：一様 0.4 Solar (必要に応じて変更可) ----
        f_Z  = lambda r, Z0=0.4: Z0 + 0.0 * r

        profiles[name] = {
            "r"    : Rcen,
            "ne"   : ne,
            "kt"   : Tx,
            "f_ne" : f_ne,
            "f_kt" : f_kt,
            "ne_i" : ne_i,
            "kt_i" : kt_i,
            "f_Z"  : f_Z,
        }

    return profiles

def plottest():
    prof = read_profile_unique('all_profiles.dat', target_names=['ABELL_0478'])
    print(prof)
    import matplotlib.pyplot as plt
    r = np.linspace(0.1, 100, 1000)
    plt.plot(r, prof['ABELL_0478']['f_ne'](r))
    plt.semilogx()
    plt.semilogy()
    plt.show()

def make_tau_list(save_path):
    
    unique_names, interpolated_ne, interpolated_kt, ne_data, kt_data, rad_data, f_Z = read_profile(file_path)
    from Resonance_Scattering_Simulation import Physics
    P = Physics()
    tau_list = []
    for i in range(0, len(unique_names)):
        tau = P.integrated_tau_E0(interpolated_ne[i], interpolated_kt[i], f_Z[i], 100, debug_mode=False)
        tau_list.append((unique_names[i], tau))
    result_df = pd.DataFrame(tau_list, columns=["name", "tau"])
    result_df.to_csv(save_path, index=False)
    return tau_list

def read_tau_list(file_path):
    # ファイルを読み込む
    data = pd.read_csv(file_path, comment='#', header=0)
    print(data)
    # 各天体ごとにデータを抽出して処理
    unique_names = data['name'].unique()
    tau_list = []
    for name in unique_names:
        # 名前ごとにデータをフィルタリング
        group = data[data['name'] == name]
        tau_list.append(group['tau'].values[0])
    return unique_names, tau_list

def plot_tau(file_path):
    from Resonance_Scattering_Simulation import RadiationField
    R = RadiationField()
    R.plot_style("single")
    thresh = 0.8
    unique_names, tau_list = read_tau_list(file_path)
    lim_unique_names = unique_names
    bins_edge = np.linspace(0, 10, 100)
    #plt.hist(tau_list, bins=bins_edge, histtype="step")
    tau_list_arr = np.array(tau_list)
    print(lim_unique_names[tau_list_arr > thresh])
    sort_tau_idx = np.argsort(tau_list_arr)
    sort_tau = np.sort(tau_list_arr)
    sort_name = lim_unique_names[sort_tau_idx]
    #plt.clf()
    R.ax.plot(sort_tau[sort_tau>thresh], 'o')
    R.ax.set_xticks(np.arange(0, len(sort_tau[sort_tau>thresh]), 1), sort_name[sort_tau>thresh], rotation=90)
    R.ax.axhline(1, color='r', linestyle='--')
    R.ax.set_ylabel(r"$\tau_{RS}$")
    pks_index = np.where(sort_name[sort_tau>thresh] == "PKS_0745-191")[0]
    perseus_index = np.where(sort_name[sort_tau>thresh] == "ABELL_0426")[0]
    centaurus_index = np.where(sort_name[sort_tau>thresh] == "CENTAURUS")[0]
    abell_2029_index = np.where(sort_name[sort_tau>thresh] == "ABELL_2029")[0]
    R.ax.axvline(abell_2029_index, color='black', linestyle='--')
    R.ax.axvline(pks_index, color='blue', linestyle='--')
    R.ax.axvline(perseus_index, color='blue', linestyle='--')
    R.ax.axvline(centaurus_index, color='blue', linestyle='--')
    R.ax.set_yscale('log')
    R.ax.tick_params(axis='x', which='major', labelsize=8)
    R.fig.tight_layout()
    plt.show()
    R.fig.savefig("tau_rs.png", dpi=300)
    prom_name = lim_unique_names[sort_tau_idx[sort_tau>thresh]]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import re

def compute_flux(Lbol, z):
    DL = cosmo.luminosity_distance(z).to(u.cm).value
    return Lbol / (4 * np.pi * DL**2)

def main(accept_path="accept_main.tab", tau_path="tau_test.csv"):
    # === データ読み込み ===
    columns = [
        "Name", "RA", "Dec", "z", "K0", "K100", "alpha", "Tcl",
        "Lbol", "Lbol_UL", "LHa", "LHa_UL", "Lrad"
    ]
    df_accept = pd.read_fwf(accept_path, names=columns, header=None, comment="#")
    df_accept["name"] = df_accept["Name"]

    # nameがABELL_123の形式なら ABELL_0123 に変換
    def format_name(name: str) -> str:
        match = re.match(r"ABELL_(\d+)$", name.upper())  # 大文字化して統一
        if match:
            num_str = match.group(1).zfill(4)
            return f"ABELL_{num_str}"
        return name

    df_accept["name"] = df_accept["name"].apply(format_name)

    df_tau = pd.read_csv(tau_path)
    df = pd.merge(df_accept, df_tau, on="name", how="inner")
    df["flux"] = compute_flux(df["Lbol"], df["z"])

    # === PKS_0745-191 の参照値 ===
    ref_name = "PKS_0745-191"
    ref = df[df["name"] == ref_name]
    if ref.empty:
        print(f"{ref_name} not found.")
        return

    ref_tau = 0.7
    ref_tau_pks = ref["tau"].values[0]
    ref_flux = ref["flux"].values[0]

    # === 条件を満たす天体（tau > PKS & flux > PKS） ===
    selected = df[(df["tau"] > ref_tau)]

    # === プロット ===
    plt.figure(figsize=(10, 7))
    plt.scatter(df["tau"], df["flux"], alpha=0.6, label="Clusters")
    #plt.yscale("log")

    # PKS を赤で強調
    plt.scatter(ref_tau_pks, ref_flux, color="red", s=100, edgecolor="black", label=ref_name)
    #plt.annotate(ref_name, (ref_tau_pks, ref_flux), textcoords="offset points", xytext=(10, 5), fontsize=9)

    # Perseus (ABELL_0426) を青で強調
    if "ABELL_0426" in df["name"]:
        perseus = df[df["name"] == "ABELL_0426"]
        plt.scatter(perseus["tau"], perseus["flux"], color="blue", s=100, edgecolor="black", label="ABELL_0426")
        for _, row in perseus.iterrows():
            plt.annotate("ABELL_0426", (row["tau"], row["flux"]), textcoords="offset points", xytext=(10, 5), fontsize=9)
    print(df["name"][:30])
    # 条件を満たす天体の名前を表示
    for _, row in selected.iterrows():
        plt.annotate(row["name"], (row["tau"], row["flux"]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel(r"$\tau_{RS}$")
    plt.ylabel("Flux [erg/s/cm²]")
    #plt.title("tau vs Flux")
    plt.grid(linestyle='dashed')
    plt.semilogy()
    #plt.legend()
    plt.tight_layout()
    plt.savefig("tau_flux.png", dpi=300)
    plt.show()


import re

# ===== 補助関数 =====

def format_name(name: str) -> str:
    """ABELL番号の整形：ABELL_123 → ABELL_0123"""
    match = re.match(r"ABELL_(\d+)$", name.upper())
    if match:
        num_str = match.group(1).zfill(4)
        return f"ABELL_{num_str}"
    return name.upper()

# ===== メイン関数 =====

def main2(accept_path="accept_main.tab", tau_path="tau_test.csv", cluster_csv_path="/Users/keitatanaka/Dropbox/AO2_proposal/analysis/cluster_catalog/Edge_1990_flux/cluster_data_full_mod.csv"):
    '''
    For AO2 proposal Figure 1 right
    '''
    accept_columns = [
        "Name", "RA", "Dec", "z", "K0", "K100", "alpha", "Tcl",
        "Lbol", "Lbol_UL", "LHa", "LHa_UL", "Lrad"
    ]
    df_accept = pd.read_fwf(accept_path, names=accept_columns, header=None, comment="#")
    df_accept["name"] = df_accept["Name"].apply(format_name)

    df_tau = pd.read_csv(tau_path)
    df_tau["name"] = df_tau["name"].apply(format_name)

    df_cluster = pd.read_csv(cluster_csv_path)
    print(df_cluster.columns)
    # df_cluster["Cluster_mod"] = df_cluster["Cluster"].apply(lambda x: x.upper().replace("ABELL ", "ABELL_").replace("A", "ABELL_") if "ABELL" in x.upper() else x.upper())
    #df_cluster["Cluster_mod"] = df_cluster["Cluster_mod"].apply(format_name)

    # --- マージ処理 ---
    df = pd.merge(df_accept, df_tau, on="name", how="inner")
    df = pd.merge(df, df_cluster[["Cluster mod", "2-10 keV Flux (10^-11 erg cm^-2 s^-1)"]], left_on="name", right_on="Cluster mod", how="inner")

    # --- Flux整形 ---
    df["flux"] = pd.to_numeric(df["2-10 keV Flux (10^-11 erg cm^-2 s^-1)"], errors="coerce")

    # === 基準天体 ===
    ref_name = "PKS_0745-191"
    ref_tau = 0.7  # デフォルト値

    ref = df[df["name"] == ref_name]
    if not ref.empty:
        ref_flux = ref["flux"].values[0]
        ref_tau = ref["tau"].values[0]
    else:
        ref_flux = None

    # ref_flux = 10
    # ref_tau = 1.755

    ref_flux = 0
    ref_tau = 0

    from Resonance_Scattering_Simulation import PlotManager
    P = PlotManager(figsize=(4,3),label_size=15)
    s = 30
    # === プロット ===
    AO1_or_PV_list = ["ABELL_2029", "CENTAURUS", "ABELL_0426", "ABELL_1795", "CYGNUS_A", "HYDRA_A", "ABELL_1060", "ABELL_1060", "ABELL_2199", "ABELL_3571", "OPHIUCHUS", "ABELL_2319"]
    target_mask = df["name"].isin(AO1_or_PV_list)
    P.axes[0].scatter(df["flux"],df["tau"],  alpha=1, label="Not Observed", color="red", s=s)
    P.axes[0].scatter(df["flux"][target_mask],df["tau"][target_mask], alpha=1, label="PV or AO1 target", color="gray", s=s)

    # 基準天体のマーカー
    # if ref_flux is not None:
    #     P.axes[0].scatter(ref_tau, ref_flux, color="red", s=100, edgecolor="black", label=ref_name)

    # Perseus 強調
    if "PKS_0745-191" in df["name"].values:
        perseus = df[df["name"] == "PKS_0745-191"]
        P.axes[0].scatter(perseus["flux"],perseus["tau"], color="blue",marker="*", s=300, edgecolor="black", label="PKS_0745-191")
    # if "ABELL_0478" in df["name"].values:
    #     perseus = df[df["name"] == "ABELL_0478"]
    #     P.axes[0].scatter( perseus["flux"],perseus["tau"], color="green",marker="*", s=100, edgecolor="black", label="ABELL 478")
        #for _, row in perseus.iterrows():
            #plt.annotate("PKS_0745-191", (row["tau"], row["flux"]), textcoords="offset points", xytext=(10, 5), fontsize=9)

    # τ > ref_tau の天体の名前を表示
    selected = df[(df["tau"] > ref_tau) | (df["flux"] > ref_flux)]

    for _, row in selected.iterrows():
        P.axes[0].annotate(row["name"], (row["flux"],row["tau"]), textcoords="offset points", xytext=(5, 5), fontsize=12)

    #P.axes[0].set_ylabel(r"Optical Depth of Fe XXV Heα w")
    #P.axes[0].set_xlabel("Flux [2-10 keV] ($10^{-11}$ erg/cm²/s)")
    P.axes[0].grid(linestyle='dashed')
    #P.axes[0].set_yscale('log')
    #P.axes[0].set_xscale('log')
    #P.axes[0].legend(loc="upper left", fontsize=12)
    P.axes[0].set_ylim(-0.5, 5.5)
    P.axes[0].set_xlim(1.25, 3e3)
    P.fig.tight_layout()
    plt.show()
    P.fig.savefig("tau_vs_flux_plot.png", dpi=300)



def main3(accept_path="accept_main.tab", tau_path="tau_test.csv", cluster_csv_path="/Users/keitatanaka/Dropbox/AO2_proposal/analysis/cluster_catalog/Edge_1990_flux/cluster_data_full_mod.csv"):
    # --- データ読み込み ---
    accept_columns = [
        "Name", "RA", "Dec", "z", "K0", "K100", "alpha", "Tcl",
        "Lbol", "Lbol_UL", "LHa", "LHa_UL", "Lrad"
    ]
    df_accept = pd.read_fwf(accept_path, names=accept_columns, header=None, comment="#")
    df_accept["name"] = df_accept["Name"].apply(format_name)

    df_tau = pd.read_csv(tau_path)
    df_tau["name"] = df_tau["name"].apply(format_name)

    df_cluster = pd.read_csv(cluster_csv_path)

    # --- マージ処理 ---
    df = pd.merge(df_accept, df_tau, on="name", how="inner")
    df = pd.merge(df, df_cluster[["Cluster mod", "2-10 keV Flux (10^-11 erg cm^-2 s^-1)"]], left_on="name", right_on="Cluster mod", how="inner")

    # --- Flux整形 ---
    df["flux"] = pd.to_numeric(df["2-10 keV Flux (10^-11 erg cm^-2 s^-1)"], errors="coerce")

    # === 基準天体 ===
    ref_name = "PKS_0745-191"
    ref_tau = 0.7  # デフォルト値

    ref = df[df["name"] == ref_name]
    if not ref.empty:
        ref_flux = ref["flux"].values[0]
        ref_tau = ref["tau"].values[0]
    else:
        ref_flux = None

    ref_flux = 0
    ref_tau = 0

    from Resonance_Scattering_Simulation import PlotManager
    P = PlotManager(figsize=(8,6),label_size=15)
    # === プロット ===
    AO1_or_PV_list = ["ABELL_2029", "CENTAURUS", "ABELL_0426", "ABELL_1795", "CYGNUS_A", "HYDRA_A", "ABELL_1060", "ABELL_1060", "ABELL_2199", "ABELL_3571", "OPHIUCHUS", "ABELL_2319"]
    target_mask = df["name"].isin(AO1_or_PV_list)
    P.axes[0].scatter(df["tau"], df["flux"], alpha=1, label="Not Observed", color="orange", s=10)
    P.axes[0].scatter(df["tau"][target_mask], df["flux"][target_mask], alpha=1, label="PV or AO1 target", color="gray", s=10)

    # 基準天体のマーカー
    if "PKS_0745-191" in df["name"].values:
        perseus = df[df["name"] == "PKS_0745-191"]
        P.axes[0].scatter(perseus["tau"], perseus["flux"], color="blue", marker="*", s=100, edgecolor="black", label="PKS_0745-191")
    if "ABELL_0478" in df["name"].values:
        perseus = df[df["name"] == "ABELL_0478"]
        P.axes[0].scatter(perseus["tau"], perseus["flux"], color="green", marker="*", s=100, edgecolor="black", label="ABELL 478")

    # τ > ref_tau の天体の名前を表示
    selected = df[(df["tau"] > ref_tau) | (df["flux"] > ref_flux)]
    for _, row in selected.iterrows():
        P.axes[0].annotate(row["name"], (row["tau"], row["flux"]), textcoords="offset points", xytext=(-5, 5), fontsize=8)

    P.axes[0].set_xlabel(r"$\tau_{RS}$")
    P.axes[0].set_ylabel("Flux [2-10 keV] ($10^{-11}$ erg/cm²/s)")
    P.axes[0].grid(linestyle='dashed')
    P.axes[0].set_yscale('log')
    P.axes[0].set_xlim(-0.5, 5)
    P.axes[0].set_ylim(1.0, 1e3)

    # --- ズームインの設定 ---
    axins = inset_axes(P.axes[0], width="50%", height="50%", loc='upper right')  # 右上に小さなインセット（ズームイン）を追加
    axins.scatter(df["tau"], df["flux"], alpha=1, color="orange", s=10)
    axins.scatter(df["tau"][target_mask], df["flux"][target_mask], alpha=1, color="gray", s=10)
    
    # ズームイン範囲の設定（例えば、特定の範囲にズーム）
    axins.set_xlim(0, 2)  # x軸のズーム範囲
    axins.set_ylim(0.1, 10)  # y軸のズーム範囲
    axins.set_yscale('log')
    
    P.fig.tight_layout()
    plt.show()
    P.fig.savefig("tau_vs_flux_plot_with_zoom.png", dpi=300)