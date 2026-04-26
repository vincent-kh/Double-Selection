# lasso2 專案 Python 重構版本

本專案是針對原始 MATLAB 變數選擇與價格風險推論流程的完整 Python 實作。在保留原始統計邏輯與控制流的基礎上，利用現代 Python 庫進行了模組化重構，並加入平行運算支援以大幅提升大規模因子運算效率。

## 核心設計目標

1. **數值對齊**：核心正規化路徑（Regularization Path）採用 `glmnet_python`，確保計算結果與原始專案依賴的 MATLAB `glmnet_matlab` 保持數值精確度的一致性。
2. **平行加速**：針對需要逐一處理大量因子的主流程與穩健性測試，整合 `joblib` 提供多核平行運算能力，顯著縮短運算時間。
3. **模組化架構**：將原本龐大的單一腳本拆解為核心算法（core）、工具函式（utils）與應用工作流（workflows）包，提升代碼的可讀性與維護性。

## 環境配置要求

### 1. 套件依賴管理
本專案使用 `uv` 作為套件管理工具。請確保系統已安裝 `uv`，並在 `python/` 目錄下執行以下命令同步虛擬環境：

```bash
uv sync
```

### 2. Fortran 運行時環境 (Runtime)
由於 `glmnet_python` 底層呼叫 Fortran 編譯的二進位檔案，在 Linux 與 macOS 系統上需要特定版本的運行庫。本專案配置為優先從 `python/` 根目錄載入 `libgfortran.so.3`。

若您的環境缺乏此函式庫，可透過 Conda 建立暫時環境並提取：

```bash
conda create -y -p python/.conda-libgfortran -c conda-forge libgfortran=3.0.0
cp python/.conda-libgfortran/lib/libgfortran.so.3.0.0 python/libgfortran.so.3
rm -rf python/.conda-libgfortran
```

## 執行指南

所有分析流程的入口腳本均位於 `python/` 根目錄。

### 主流程分析 (Main Analysis)
執行主要的因子分析與價格風險估計（對應論文 Table 1）：
```bash
uv run python/main.py [選項]
```
參數說明：
* `-n`, `--n_jobs`：平行運算使用的核心數（預設為 -1，即使用所有可用 CPU）。
* `-q`, `--quiet`：抑制運算過程中因數值邊界產生的 RuntimeWarning（建議在平行執行時使用以保持輸出整潔）。

### 穩健性測試 (Robustness Checks)
執行包含 5x5 投組、PCA 轉換及逐步選擇（Stepwise Selection）在內的系列穩定性測試：
```bash
uv run python/robustness.py [選項]
```
參數說明：
* `-n`, `--n_jobs`：平行運算核心數。

### 模擬分析 (Simulation)
執行蒙地卡羅模擬（Monte Carlo Simulation）並產出分布圖與統計偏誤分析（對應論文 Figure 11）：
```bash
uv run python/simulation.py
```

### 論文圖表重製
重製論文中的 Figure 1（各個控制因子的選擇率柱狀圖）：
```bash
uv run python/plot_figure1.py
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
