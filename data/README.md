# Data

This directory does not contain the raw dataset itself — it is excluded from
version control via `.gitignore` (see [Reproducing the Data](#reproducing-the-data)
below). This file documents where the data comes from and what happens to it
before it reaches the notebooks.

## Source

**Dataset:** Online Retail

**Dataset_link:** "https://www.kaggle.com/datasets/vijayuv/onlineretail"

**Provider:** UCI Machine Learning Repository (mirrored on Kaggle)

**Nature:** Transactional invoice-level data from a UK-based online retailer,
covering a single SKU-level purchase history for a mix of UK and international customers.

Each row represents one line item on an invoice, with the core fields being:

| Column | Description |
|---|---|
| `InvoiceNo` | Unique invoice identifier. Invoices prefixed with `C` are cancellations. |
| `StockCode` | Product/item code. |
| `Description` | Product name. |
| `Quantity` | Units purchased for that line item (negative values indicate returns). |
| `InvoiceDate` | Timestamp of the transaction. |
| `UnitPrice` | Price per unit. |
| `CustomerID` | Unique customer identifier (a meaningful share of rows have this missing). |
| `Country` | Customer's country. |

## Cleaning Pipeline (Phase 1)

The raw file is processed in `notebooks/01_data_cleaning_eda.ipynb`, which produces
the cleaned transaction table used by every downstream phase. The key steps
implemented there:

- **Missing `CustomerID` removal** — rows without a customer identifier can't be
  attributed to a customer and are dropped, since every downstream phase (RFM,
  churn, LTV) operates at the customer level.
- **Cancellation/negative-quantity handling** — cancelled orders (`InvoiceNo`
  starting with `C`) and negative `Quantity` rows are addressed as part of
  building a clean, revenue-accurate transaction table.
- **Outlier removal on both `Quantity` and `UnitPrice`** — an earlier version of
  the notebook only filtered outliers on `Quantity`; this was corrected so
  extreme `UnitPrice` values are filtered as well, since leaving them in
  silently inflated aggregate revenue figures.
- **Execution-order fix** — the notebook was restructured to eliminate a silent
  out-of-order cell execution issue that had been producing incorrect revenue
  figures in earlier runs. Re-running the notebook top-to-bottom now
  reproduces consistent numbers.
- **Revenue feature construction** — a line-level revenue field (`Quantity ×
  UnitPrice`) is derived for use in RFM and LTV calculations downstream.

## Pipeline Flow

```
data/raw/<online_retail_source_file>
        │
        ▼
01_data_cleaning_eda.ipynb   →  cleaned transaction table (Parquet)
        │
        ▼
02_rfm_segmentation.ipynb    →  customer-level RFM features + segment labels
        │
        ▼
03_churn_prediction.ipynb    →  churn model inputs (rebuilt from raw transactions
        │                        per customer, not reused from RFM, to avoid
        │                        label leakage)
        ▼
04_ltv_modeling.ipynb        →  BG/NBD + Gamma-Gamma inputs (frequency/recency/T,
                                 monetary value per the `lifetimes` library's
                                 definitions)
```

Intermediate outputs are cached as Parquet files between phases rather than
re-computed in every notebook, since re-deriving RFM/BG-NBD inputs from raw
transactions is nontrivial and reused downstream.

## A Note on Feature Definitions

The `lifetimes` library's definitions of `frequency`, `recency`, and `T` differ
from the standard RFM table definitions used in Phase 2. These are computed
separately and should not be conflated — a feature table built for RFM
segmentation is not a valid direct input to the BG/NBD model.

## Reproducing the Data

1. Download the Online Retail dataset from Kaggle/UCI.
2. Place the file in `data/raw/`.
3. Run `notebooks/01_data_cleaning_eda.ipynb` top-to-bottom to regenerate the
   cleaned Parquet file consumed by later notebooks.
