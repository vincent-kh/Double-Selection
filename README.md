# Double Selection

本程式是針對原始 MATLAB 變數選擇與價格風險推論流程的完整 Python 實作。在保留原始統計邏輯與控制流的基礎上，利用現代 Python 庫進行了模組化重構，並加入平行運算支援以大幅提升大規模因子運算效率。

## 核心設計目標

1. **數值對齊**：核心正規化路徑（Regularization Path）採用 `glmnet_python`，確保計算結果與原始專案依賴的 MATLAB `glmnet_matlab` 保持數值精確度的一致性。
2. **平行加速**：針對需要逐一處理大量因子的主流程與穩健性測試，整合 `joblib` 提供多核平行運算能力，顯著縮短運算時間。
3. **模組化架構**：將原本龐大的單一腳本拆解為核心算法（core）、工具函式（utils）與應用工作流（workflows）包，提升代碼的可讀性與維護性。

## 環境配置要求

> **注意：** 本專案核心依賴 `glmnet_python` 的 Fortran 編譯版本，**僅限定於 Linux 環境執行**（推薦使用 Ubuntu 22.04+ 或 Windows WSL2）。

### 1. 安裝與設定環境 (Linux/WSL2)

#### 安裝 uv
本專案使用 [uv](https://github.com/astral.sh/uv) 進行環境與依賴管理：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 建立虛擬環境
進入 `python/` 目錄並執行：
```bash
cd python
uv sync
```

### 2. Fortran 運行時環境 (Runtime)
由於 `glmnet_python` 底層呼叫 Fortran 編譯的二進位檔案，本專案已在 `python/` 根目錄預置了適用於 Linux 的 `libgfortran.so.3`。

若您的系統環境無法識別該檔案，可透過 Conda 重新提取對應 Linux 版本的函式庫：

```bash
conda create -y -p python/.conda-libgfortran -c conda-forge libgfortran=3.0.0
cp python/.conda-libgfortran/lib/libgfortran.so.3.0.0 python/libgfortran.so.3
rm -rf python/.conda-libgfortran
```

## 執行指南

所有分析流程的入口腳本均位於 `python/` 根目錄。請確保在 `python/` 目錄下執行命令。

### 主流程分析 (Main Analysis)
執行主要的因子分析與價格風險估計（對應論文 Table 1）：
```bash
uv run main.py [選項]
```
參數說明：
* `-n`, `--n_jobs`：平行運算使用的核心數（預設為 -1）。
* `-q`, `--quiet`：抑制運算過程中的 RuntimeWarning。

### 穩健性測試 (Robustness Checks)
執行系列穩定性測試：
```bash
uv run robustness.py [選項]
```

### 模擬分析 (Simulation)
執行蒙地卡羅模擬（對應論文 Figure 11）：
```bash
uv run simulation.py
```

### 論文圖表重製
重製論文中的 Figure 1（選擇率柱狀圖）：
```bash
uv run plot_figure1.py
```

## 專案目錄結構

* **python/lasso2/**：核心套件目錄。
    * `core.py`：雙選擇（DS）與 OLS 核心算法實作。
    * `utils.py`：路徑常量、PCA 協方差計算與通用數值工具。
    * `selection.py`：時間序列交叉驗證與逐步回歸邏輯。
    * `inference.py`：價格風險推論與標準誤計算。
    * `workflows/`：拆分後的各項具體分析流程模組。
* **python/data/**：集中存放所有輸入資料（`.csv` 與 `.mat`）。
* **python/csv/**：統一儲存所有分析結果的 CSV 數據表。

## 資料管理說明
為了優化執行效率與模組解耦，所有原本散落在 MATLAB 各個子目錄的原始資料已統一遷移至 `python/data/`。所有 Python 腳本會自動讀取該目錄，並將計算結果輸出至 `python/csv/`。

## TODO
- [ ] 支援台股資料分析 (Taiwan Stock Market Analysis)

