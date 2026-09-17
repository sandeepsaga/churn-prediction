# app.py
from pathlib import Path
import streamlit as st


# Safe imports with visible error logging
try:
    import joblib
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use('Agg') # Prevents GUI crashes on Linux servers
    import matplotlib.pyplot as plt
    import shap
except Exception as e:
    st.error(f"Error loading dependencies: {e}")
    st.stop()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'

if not DATA_DIR.exists():
    st.error(f"Data directory missing on server: {DATA_DIR}")
if not MODEL_DIR.exists():
    st.error(f"Model directory missing on server: {MODEL_DIR}")

st.set_page_config(page_title="Customer Analytics Dashboard", page_icon="📊", layout="wide")

SEGMENT_COLORS = {
    'Champions': '#22C55E',
    'At Risk': '#F59E0B',
    'Hibernating': '#6B7280',
    'New Customers': '#3B82F6',
}
RISK_COLORS = {'Low Risk': '#22C55E', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'}
CHART_TEMPLATE = 'plotly_white'


def kpi_card(col, icon, label, value, caption=None):
    """Custom KPI card using st.container(border=True) -- a built-in, version-stable
    Streamlit feature, rather than CSS targeting internal (version-fragile) testids."""
    with col:
        with st.container(border=True):
            st.markdown(f"<div style='font-size:13px;color:#6B7280;font-weight:600;'>"
                        f"{icon} {label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:30px;font-weight:800;color:#111827;"
                        f"margin-top:2px;'>{value}</div>", unsafe_allow_html=True)
            if caption:
                st.markdown(f"<div style='font-size:12px;color:#9CA3AF;'>{caption}</div>",
                            unsafe_allow_html=True)


def segment_badge(segment: str) -> str:
    color = SEGMENT_COLORS.get(segment, '#9CA3AF')
    return (f'<span style="background-color:{color};color:white;font-weight:600;'
            f'padding:4px 14px;border-radius:20px;font-size:14px;">{segment}</span>')


# ==========================================
# DATA & MODEL LOADING (cached)
# ==========================================
@st.cache_data
def load_data():
    rfm = pd.read_parquet(DATA_DIR / 'rfm_segmented.parquet')
    churn = pd.read_parquet(DATA_DIR / 'churn_predictions.parquet')
    ltv = pd.read_parquet(DATA_DIR / 'ltv_matrix.parquet')
    retention = pd.read_parquet(DATA_DIR / 'retention_matrix.parquet')

    # Defensive loop: Catch if CustomerID is hiding in the index or is lowercase
    for name, d in [('RFM', rfm), ('Churn', churn), ('LTV', ltv)]:
        if 'CustomerID' not in d.columns:
            if d.index.name == 'CustomerID' or 'CustomerID' in d.index.names:
                d.reset_index(inplace=True)
            elif 'customer_id' in d.columns:
                d.rename(columns={'customer_id': 'CustomerID'}, inplace=True)
            else:
                st.error(f"🚨 Missing 'CustomerID' in {name} dataset. Found columns: {d.columns.tolist()}")
                st.stop()
        
        # Now it is safe to convert
        d['CustomerID'] = d['CustomerID'].astype(str)

    ltv_cols = [c for c in ['CustomerID', 'LTV_3m', 'LTV_6m', 'LTV_12m'] if c in ltv.columns]
    master = rfm.merge(churn, on='CustomerID', how='left') \
                .merge(ltv[ltv_cols], on='CustomerID', how='left')
    return master, retention

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / 'xgb_churn_model.pkl')
    feature_names = joblib.load(MODEL_DIR / 'model_features.pkl')
    return model, feature_names


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


master_df, retention_df = load_data()
xgb_model, model_feature_cols = load_model()
explainer = get_explainer(xgb_model)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 📊 Dashboard Info")
    st.divider()
    st.metric("Total Customers", f"{master_df['CustomerID'].nunique():,}")
    st.metric("Segments Identified", master_df['Segment'].nunique())
    st.divider()
    st.caption("**Pipeline:** RFM Segmentation → Churn Prediction (XGBoost) → "
               "Probabilistic LTV (BG/NBD + Gamma-Gamma)")
    st.caption("**Data:** Online Retail Dataset (Kaggle/UCI)")


# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div style="background: linear-gradient(90deg,#4F46E5,#7C3AED); padding: 28px 32px;
            border-radius: 14px; margin-bottom: 24px;">
  <div style="color:white; font-size:28px; font-weight:800;">📊 Customer Analytics Dashboard</div>
  <div style="color:#E0E7FF; font-size:15px; margin-top:6px;">
    RFM Segmentation · Churn Prediction · Probabilistic Lifetime Value
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈  Executive Insights", "🎯  Strategic Segments", "🔍  Customer Lookup"])


# ------------------------------------------
# TAB 1: EXECUTIVE INSIGHTS
# ------------------------------------------
with tab1:
    total_revenue = master_df['Monetary'].sum()
    avg_ltv = master_df['LTV_12m'].mean()
    churn_rate = (master_df['churn_probability'] >= 0.5).mean()
    total_customers = master_df['CustomerID'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "💰", "Total Revenue", f"${total_revenue:,.0f}")
    kpi_card(c2, "📈", "Avg. 12-Month LTV", f"${avg_ltv:,.0f}", "Repeat customers only")
    kpi_card(c3, "⚠️", "Predicted Churn Rate", f"{churn_rate:.1%}", "Probability ≥ 50%")
    kpi_card(c4, "👥", "Total Customers", f"{total_customers:,}")

    st.write("")
    with st.container(border=True):
        st.subheader("Customer Retention by Monthly Cohort")

        n_rows, n_cols = retention_df.shape
        fig_height = max(420, 90 + n_rows * 38)

        z = retention_df.values
        text = [[f"{val:.1%}" if pd.notna(val) else "" for val in row] for row in z]

        fig = go.Figure(data=go.Heatmap(
            z=z, x=retention_df.columns.astype(str), y=retention_df.index.astype(str),
            text=text, texttemplate="%{text}", textfont={"size": 11},
            colorscale='Blues', zmin=0, zmax=0.5, xgap=2, ygap=2,
            colorbar=dict(title="Retention", tickformat='.0%')
        ))
        fig.update_layout(
            template=CHART_TEMPLATE, height=fig_height,
            xaxis_title="Months Since First Purchase", yaxis_title="Cohort Month",
            yaxis=dict(autorange='reversed'), xaxis=dict(dtick=1),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------
# TAB 2: STRATEGIC SEGMENTS
# ------------------------------------------
with tab2:
    ADVICE = {
        'Champions': "Reward loyalty — early access to new products, referral incentives.",
        'At Risk': "Valuable customers going quiet. Trigger a personalized win-back offer now.",
        'Hibernating': "Low historical value and inactive. Low-cost re-engagement only.",
        'New Customers': "Nurture with onboarding content and a second-purchase incentive.",
    }

    with st.container(border=True):
        st.subheader("Filters")
        available_segments = sorted(master_df['Segment'].dropna().unique())
        selected_segments = st.multiselect("Segment", available_segments, default=available_segments)
        min_risk, max_risk = st.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

    filtered = master_df[
        master_df['Segment'].isin(selected_segments) &
        master_df['churn_probability'].between(min_risk, max_risk)
    ]
    st.write(f"**{filtered.shape[0]:,} customers match this filter**")

    for seg in selected_segments:
        color = SEGMENT_COLORS.get(seg, '#9CA3AF')
        st.markdown(
            f'<div style="border-left:5px solid {color}; border-radius:6px; padding:10px 16px;'
            f'margin-bottom:8px; background-color:#FAFAFA;"><b>{seg}:</b> '
            f'{ADVICE.get(seg, "No advice defined yet.")}</div>',
            unsafe_allow_html=True
        )

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            seg_ltv = master_df.groupby('Segment')['LTV_12m'].mean().reset_index()
            fig_ltv = px.bar(seg_ltv, x='Segment', y='LTV_12m', color='Segment',
                              color_discrete_map=SEGMENT_COLORS, text_auto='.2s',
                              title="Avg. Predicted 12-Month LTV by Segment")
            fig_ltv.update_traces(texttemplate='$%{text}', textposition='outside')
            fig_ltv.update_layout(template=CHART_TEMPLATE, showlegend=False, yaxis_title="Avg. LTV ($)")
            st.plotly_chart(fig_ltv, use_container_width=True)

    with col2:
        with st.container(border=True):
            seg_churn = master_df.groupby('Segment')['churn_probability'].mean().reset_index()
            fig_churn = px.bar(seg_churn, x='Segment', y='churn_probability', color='Segment',
                                color_discrete_map=SEGMENT_COLORS, text_auto='.1%',
                                title="Avg. Churn Probability by Segment")
            fig_churn.update_traces(textposition='outside')
            fig_churn.update_layout(template=CHART_TEMPLATE, showlegend=False,
                                     yaxis_title="Avg. Churn Probability", yaxis_tickformat='.0%')
            st.plotly_chart(fig_churn, use_container_width=True)

    st.write("")
    with st.container(border=True):
        st.subheader("Customer Detail")
        display_df = filtered[['CustomerID', 'Segment', 'Recency', 'Frequency',
                                'Monetary', 'churn_probability', 'LTV_12m']].rename(columns={
            'Recency': 'Recency (days)', 'Frequency': 'Frequency (orders)',
            'Monetary': 'Total Spend', 'churn_probability': 'Churn Risk', 'LTV_12m': '12-Month LTV'
        })
        styled = display_df.style.format({
            'Total Spend': '${:,.2f}',
            'Churn Risk': '{:.1%}',
            '12-Month LTV': lambda x: f"${x:,.0f}" if pd.notna(x) else "—",
            'Recency (days)': '{:,.0f}',
            'Frequency (orders)': '{:,.0f}',
        }).bar(subset=['Churn Risk'], color='#FCA5A5', vmin=0, vmax=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download filtered customer list (CSV)",
                            filtered.to_csv(index=False).encode('utf-8'),
                            file_name="segment_export.csv", mime="text/csv")


# ------------------------------------------
# TAB 3: CUSTOMER LOOKUP
# ------------------------------------------
with tab3:
    st.subheader("Search a Customer")
    st.caption("See their segment, churn risk, predicted 12-month LTV, and the specific "
               "factors driving that prediction.")

    if 'customer_search' not in st.session_state:
        st.session_state.customer_search = ''

    def _set_example(cid):
        st.session_state.customer_search = cid

    st.text_input("Customer ID", placeholder="e.g. 17850", key='customer_search')

    st.caption("Or try an example:")
    examples = [("12347", "Champion"), ("14646", "Top LTV"),
                ("12349", "New / One-time"), ("12350", "Hibernating")]
    ex_cols = st.columns(4)
    for col, (eid, label) in zip(ex_cols, examples):
        col.button(f"{eid} · {label}", on_click=_set_example, args=(eid,), use_container_width=True)

    customer_id_input = st.session_state.customer_search

    if not customer_id_input:
        st.write("")
        with st.container(border=True):
            st.markdown("### 👋 No customer selected yet")
            st.write(
                f"This dashboard covers **{master_df['CustomerID'].nunique():,} customers**. "
                "Enter any Customer ID above, or click an example, to see their full profile — "
                "segment, churn risk, predicted lifetime value, and a SHAP breakdown of exactly "
                "why the model flagged that risk level."
            )
    else:
        customer_row = master_df[master_df['CustomerID'] == customer_id_input.strip()]

        if customer_row.empty:
            st.warning(f"No customer found with ID '{customer_id_input}'. Try one of the examples above.")
        else:
            row = customer_row.iloc[0]

            with st.container(border=True):
                st.markdown(f"### Customer {row['CustomerID']} &nbsp; {segment_badge(row['Segment'])}",
                            unsafe_allow_html=True)
                st.write("")
                c1, c2, c3 = st.columns(3)
                kpi_card(c1, "🕒", "Recency", f"{row['Recency']:.0f} days")

                risk = row['churn_probability']
                risk_label = "High Risk" if risk >= 0.7 else "Medium Risk" if risk >= 0.4 else "Low Risk"
                kpi_card(c2, "⚠️", "Churn Probability", f"{risk:.1%}", risk_label)

                if pd.isna(row['LTV_12m']):
                    kpi_card(c3, "💰", "Predicted 12-Month LTV", "N/A", "One-time buyer")
                else:
                    kpi_card(c3, "💰", "Predicted 12-Month LTV", f"${row['LTV_12m']:,.0f}")

            st.write("")
            with st.container(border=True):
                st.subheader("Why this prediction? (SHAP Waterfall)")
                customer_features = customer_row[model_feature_cols]
                shap_values = explainer(customer_features)

                base_value = explainer.expected_value
                if hasattr(base_value, "__len__"):
                    base_value = base_value[0]
                sv = shap_values.values
                if sv.ndim == 3:
                    sv = sv[:, :, 1]

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap.Explanation(
                    values=sv[0], base_values=base_value,
                    data=customer_features.iloc[0], feature_names=customer_features.columns.tolist()
                ), show=False)
                st.pyplot(fig)
                plt.close(fig)

st.divider()
st.caption("Built with Streamlit · XGBoost · SHAP · lifetimes")