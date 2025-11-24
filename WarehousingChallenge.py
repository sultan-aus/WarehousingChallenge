import time
import io
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ========= CONFIG =========

st.set_page_config(
    page_title="Warehouse Activity Profiling Simulator",
    layout="wide"
)

ABC_THRESHOLDS = {"A": 0.80, "B": 0.95, "C": 1.00}
DEFAULT_N_CLUSTERS = 4
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.1

DEFAULT_LOCAL_DATASET = (
    Path.home() / "Downloads" / "Warehouse_Synthetic_Dataset_2025.xlsx"
)


# ========= CORE HELPERS =========

def load_excel(file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Orders, Lines, SKU_Master, Storage_Zones from Excel file or file-like."""
    xls = pd.ExcelFile(file)
    required_sheets = ["Orders", "Lines", "SKU_Master", "Storage_Zones"]
    missing = [s for s in required_sheets if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"Missing sheets: {missing}")

    orders = pd.read_excel(xls, "Orders")
    lines = pd.read_excel(xls, "Lines")
    sku_master = pd.read_excel(xls, "SKU_Master")
    storage_zones = pd.read_excel(xls, "Storage_Zones")
    return orders, lines, sku_master, storage_zones


def validate_relations(orders, lines, sku_master, storage_zones) -> List[str]:
    errors = []

    # Lines → Orders
    missing_orders = set(lines["Order_ID"].unique()) - set(orders["Order_ID"].unique())
    if missing_orders:
        errors.append(f"Lines reference unknown Order_IDs: {list(missing_orders)[:5]} ...")

    # Lines → SKU_Master
    missing_skus = set(lines["SKU_ID"].unique()) - set(sku_master["SKU_ID"].unique())
    if missing_skus:
        errors.append(f"Lines reference unknown SKU_IDs: {list(missing_skus)[:5]} ...")

    # Storage types
    sku_types = set(sku_master["Storage_Type"].unique())
    zone_types = set(storage_zones["Storage_Type"].unique())
    missing_types = sku_types - zone_types
    if missing_types:
        errors.append(f"No storage zones defined for Storage_Type(s): {missing_types}")

    return errors


def demand_analysis(sku_master: pd.DataFrame) -> pd.DataFrame:
    df = sku_master.copy()
    df = df.sort_values("Demand", ascending=False).reset_index(drop=True)

    total_demand = df["Demand"].sum()
    df["Demand_Share"] = df["Demand"] / total_demand
    df["Cum_Demand_Share"] = df["Demand_Share"].cumsum()

    def classify_abc(cum_share):
        if cum_share <= ABC_THRESHOLDS["A"]:
            return "A"
        elif cum_share <= ABC_THRESHOLDS["B"]:
            return "B"
        else:
            return "C"

    df["ABC_Class"] = df["Cum_Demand_Share"].apply(classify_abc)
    return df


def clustering_skus(sku_df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS):
    df = sku_df.copy()
    features = df[["Demand", "Weight_kg", "Volume_m3"]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    return df, kmeans, scaler


def build_basket_matrix(lines: pd.DataFrame) -> pd.DataFrame:
    order_sku_lists = (lines.groupby("Order_ID")["SKU_ID"]
                       .apply(list)
                       .reset_index())
    transactions = order_sku_lists["SKU_ID"].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    return basket_df


def association_mining(lines: pd.DataFrame,
                       min_support: float = MIN_SUPPORT,
                       min_confidence: float = MIN_CONFIDENCE,
                       min_lift: float = MIN_LIFT) -> pd.DataFrame:
    basket = build_basket_matrix(lines)
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets,
                              metric="confidence",
                              min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return pd.DataFrame()

    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )

    # Handle different column names in different mlxtend versions
    cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    
    # Add support columns if they exist (column names vary by mlxtend version)
    if "antecedent support" in rules.columns:
        cols.extend(["antecedent support", "consequent support"])
    elif "antecedent_support" in rules.columns:
        cols.extend(["antecedent_support", "consequent_support"])
    
    return rules[cols].sort_values(["lift", "confidence"], ascending=False)


def prepare_graph_edges_from_rules(rules: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    rules_top = rules.sort_values("lift", ascending=False).head(top_n)
    edge_rows = []
    for _, row in rules_top.iterrows():
        ants = row["antecedents_str"].split(", ")
        cons = row["consequents_str"].split(", ")
        for a in ants:
            for c in cons:
                edge_rows.append({"source": a, "target": c, "weight": row["lift"]})
    return pd.DataFrame(edge_rows)


def slot_skus(sku_master: pd.DataFrame, storage_zones: pd.DataFrame):
    """Demand-based slotting heuristic."""
    sku_df = sku_master.copy()
    zones_df = storage_zones.copy()
    zones_df["Remaining_Capacity_m3"] = zones_df["Capacity_m3"].astype(float)

    assignment_rows = []
    storage_types = sku_df["Storage_Type"].unique()

    for st in storage_types:
        skus_st = (sku_df[sku_df["Storage_Type"] == st]
                   .sort_values("Demand", ascending=False)
                   .reset_index(drop=True))

        zones_st_idx = zones_df["Storage_Type"] == st
        zones_st = (zones_df[zones_st_idx]
                    .sort_values("Distance", ascending=True)
                    .reset_index(drop=True))

        if zones_st.empty:
            for _, srow in skus_st.iterrows():
                assignment_rows.append({
                    "SKU_ID": srow["SKU_ID"],
                    "Demand": srow["Demand"],
                    "Volume_m3": srow["Volume_m3"],
                    "Storage_Type": st,
                    "Assigned_Zone": "UNASSIGNED",
                    "Zone_Distance": np.nan
                })
            continue

        for _, srow in skus_st.iterrows():
            sku_id = srow["SKU_ID"]
            vol = float(srow["Volume_m3"])
            demand = float(srow["Demand"])

            assigned_zone_id = None
            assigned_distance = np.nan

            for _, zrow in zones_st.iterrows():
                zid = zrow["Zone_ID"]
                mask_zone = zones_df["Zone_ID"] == zid
                remaining_cap = float(
                    zones_df.loc[mask_zone, "Remaining_Capacity_m3"].iloc[0]
                )
                if vol <= remaining_cap + 1e-9:
                    assigned_zone_id = zid
                    assigned_distance = float(zrow["Distance"])
                    zones_df.loc[mask_zone, "Remaining_Capacity_m3"] = remaining_cap - vol
                    break

            if assigned_zone_id is None:
                assigned_zone_id = "UNASSIGNED"

            assignment_rows.append({
                "SKU_ID": sku_id,
                "Demand": demand,
                "Volume_m3": vol,
                "Storage_Type": st,
                "Assigned_Zone": assigned_zone_id,
                "Zone_Distance": assigned_distance
            })

    assignments_df = pd.DataFrame(assignment_rows)

    used_vol_per_zone = (
        assignments_df[assignments_df["Assigned_Zone"] != "UNASSIGNED"]
        .groupby("Assigned_Zone")["Volume_m3"]
        .sum()
        .rename("Used_Volume_m3")
    )

    zone_report_df = zones_df[["Zone_ID", "Storage_Type", "Capacity_m3"]].copy()
    zone_report_df = zone_report_df.set_index("Zone_ID").join(used_vol_per_zone, how="left")
    zone_report_df["Used_Volume_m3"] = zone_report_df["Used_Volume_m3"].fillna(0.0)
    zone_report_df["Remaining_Volume_m3"] = (
        zone_report_df["Capacity_m3"] - zone_report_df["Used_Volume_m3"]
    )
    zone_report_df["Utilization_Percent"] = np.where(
        zone_report_df["Capacity_m3"] > 0,
        100.0 * zone_report_df["Used_Volume_m3"] / zone_report_df["Capacity_m3"],
        0.0
    )
    zone_report_df = zone_report_df.reset_index()

    valid_assignments = assignments_df[assignments_df["Assigned_Zone"] != "UNASSIGNED"].copy()
    valid_assignments["Demand_Weighted_Distance"] = (
        valid_assignments["Demand"] * valid_assignments["Zone_Distance"]
    )
    total_dwd = valid_assignments["Demand_Weighted_Distance"].sum()

    return assignments_df, zone_report_df, total_dwd


def adjust_for_scenario(sku_master: pd.DataFrame,
                        storage_zones: pd.DataFrame,
                        demand_multiplier: float = 1.0,
                        capacity_multiplier: float = 1.0):
    sku_adj = sku_master.copy()
    zones_adj = storage_zones.copy()
    sku_adj["Demand"] = (sku_adj["Demand"] * demand_multiplier).round().astype(int)
    zones_adj["Capacity_m3"] = zones_adj["Capacity_m3"] * capacity_multiplier
    return sku_adj, zones_adj


def plot_zone_utilization(zone_report_df: pd.DataFrame):
    df = zone_report_df.copy().sort_values("Zone_ID")
    fig = px.bar(
        df,
        x="Zone_ID",
        y="Utilization_Percent",
        color="Storage_Type",
        title="Storage Zone Utilization (%)",
        labels={"Utilization_Percent": "Utilization (%)", "Zone_ID": "Zone"}
    )
    fig.update_layout(
        yaxis=dict(range=[0, max(100, df["Utilization_Percent"].max() + 10)])
    )
    return fig


def plot_skus_per_zone_by_demand_level(assignments_df: pd.DataFrame,
                                       sku_df: pd.DataFrame):
    sku_info = sku_df.copy()
    if "ABC_Class" not in sku_info.columns:
        sku_info = demand_analysis(sku_info)

    merged = assignments_df.merge(
        sku_info[["SKU_ID", "ABC_Class"]],
        on="SKU_ID",
        how="left"
    )
    merged = merged[merged["Assigned_Zone"] != "UNASSIGNED"].copy()

    counts = (merged
              .groupby(["Assigned_Zone", "ABC_Class"])["SKU_ID"]
              .nunique()
              .reset_index(name="SKU_Count"))

    pivot = counts.pivot(index="Assigned_Zone",
                         columns="ABC_Class",
                         values="SKU_Count").fillna(0)

    demand_levels = ["A", "B", "C"]
    present = [d for d in demand_levels if d in pivot.columns]
    pivot = pivot[present].sort_index()

    fig = go.Figure()
    for level in present:
        fig.add_bar(
            x=pivot.index,
            y=pivot[level],
            name=f"Demand Level {level}"
        )
    fig.update_layout(
        barmode="stack",
        title="SKUs per Zone by Demand Level (ABC)",
        xaxis_title="Zone",
        yaxis_title="Number of SKUs"
    )
    return fig


def build_cooccurrence_graph(edges_df: pd.DataFrame):
    if edges_df.empty:
        return go.Figure()

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=12, line_width=1)
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="SKU Co-occurrence Network",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
        )
    )
    return fig


def to_excel_bytes(sheet_dict: Dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheet_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


# ========= SPLASH SCREEN =========

if "show_splash" not in st.session_state:
    st.session_state.show_splash = True

if st.session_state.show_splash:
    with st.container():
        st.markdown(
            """
            <div style="text-align:center; padding-top:60px;">
              <h1>American University of Sharjah</h1>
              <h3>Industrial Engineering Department</h3>
              <h2>Warehouse Activity Profiling Simulator</h2>
              <p>Developed for INE 494-5: Analysis of Procurement and Warehousing Operations</p>
              <br>
              <h4>Team Members:</h4>
              <p style="line-height:1.8;">
                Sultan Albinali - B00103378<br>
                Mohamad Ftouni - B00089796<br>
                Fawaz Al Jandali - B00082922<br>
                Ahmed Abusnina - B00095469
              </p>
              <br>
              <h4>Loading Slotting Simulator … Please Wait.</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    time.sleep(5)
    st.session_state.show_splash = False
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ========= SIDEBAR – DATA IMPORT & SCENARIOS =========

st.sidebar.header("Data Import & Scenarios")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel dataset (Orders, Lines, SKU_Master, Storage_Zones)",
    type=["xlsx"]
)

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

# Try to get dataset: upload preferred, else fallback to ~/Downloads/...
orders_df = lines_df = sku_master_df = storage_zones_df = None
using_fallback = False

if uploaded_file is not None:
    try:
        orders_df, lines_df, sku_master_df, storage_zones_df = load_excel(uploaded_file)
        st.sidebar.success("Dataset loaded from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded file: {e}")
elif DEFAULT_LOCAL_DATASET.exists():
    try:
        orders_df, lines_df, sku_master_df, storage_zones_df = load_excel(DEFAULT_LOCAL_DATASET)
        using_fallback = True
        st.sidebar.info(f"Using default local file: {DEFAULT_LOCAL_DATASET}")
    except Exception as e:
        st.sidebar.error(f"Error loading default local file: {e}")
else:
    st.warning(
        "Please upload an Excel dataset, or place "
        f"'Warehouse_Synthetic_Dataset_2025.xlsx' in your Downloads folder."
    )
    st.stop()

# Validate relationships
errors = validate_relations(orders_df, lines_df, sku_master_df, storage_zones_df)
if errors:
    st.error("Data validation issues detected:")
    for e in errors:
        st.write("- ", e)
    st.stop()

# ========= PRE-COMPUTE ANALYTICS =========

sku_with_abc = demand_analysis(sku_master_df)
sku_with_clusters, kmeans_model, scaler = clustering_skus(sku_with_abc)

rules_df = association_mining(lines_df)
edges_df = prepare_graph_edges_from_rules(rules_df, top_n=50)

# Baseline slotting
baseline_assignments, baseline_zone_report, baseline_dwd = slot_skus(
    sku_with_abc, storage_zones_df
)
baseline_metrics = {
    "Total_Demand_Weighted_Distance": baseline_dwd,
    "Avg_Zone_Utilization": baseline_zone_report["Utilization_Percent"].mean()
}

if "Baseline" not in st.session_state.scenarios:
    st.session_state.scenarios["Baseline"] = {
        "assignments": baseline_assignments,
        "zones": baseline_zone_report,
        "sku": sku_with_abc,
        "metrics": baseline_metrics
    }

# Scenario controls
st.sidebar.subheader("What-if Scenario Controls")
demand_multiplier = st.sidebar.slider(
    "Demand multiplier",
    min_value=0.5, max_value=2.0, value=1.0, step=0.1
)
capacity_multiplier = st.sidebar.slider(
    "Zone capacity multiplier",
    min_value=0.5, max_value=2.0, value=1.0, step=0.1
)
scenario_name = st.sidebar.text_input("Scenario name", value="High Demand")

if st.sidebar.button("Run & Save Scenario"):
    sku_adj, zones_adj = adjust_for_scenario(
        sku_with_abc,
        storage_zones_df,
        demand_multiplier=demand_multiplier,
        capacity_multiplier=capacity_multiplier
    )
    assignments_s, zones_report_s, dwd_s = slot_skus(sku_adj, zones_adj)
    metrics_s = {
        "Total_Demand_Weighted_Distance": dwd_s,
        "Avg_Zone_Utilization": zones_report_s["Utilization_Percent"].mean()
    }
    st.session_state.scenarios[scenario_name] = {
        "assignments": assignments_s,
        "zones": zones_report_s,
        "sku": sku_adj,
        "metrics": metrics_s
    }
    st.sidebar.success(f"Scenario '{scenario_name}' saved.")


# ========= MAIN TABS =========

tab_overview, tab_analytics, tab_slotting, tab_scenarios, tab_export = st.tabs(
    ["Overview", "Analytics & Clustering", "Slotting & Simulation", "Scenarios", "Export"]
)

# --- OVERVIEW TAB ---
with tab_overview:
    st.title("Warehouse Activity Profiling Simulator")

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Orders", len(orders_df))
    col2.metric("Number of SKUs", len(sku_master_df))
    col3.metric("Number of Storage Zones", len(storage_zones_df))

    st.subheader("Baseline Slotting KPIs")
    col4, col5 = st.columns(2)
    col4.metric(
        "Total Demand-Weighted Travel Distance",
        f"{baseline_dwd:,.0f}"
    )
    col5.metric(
        "Average Zone Utilization",
        f"{baseline_metrics['Avg_Zone_Utilization']:.1f}%"
    )

    if using_fallback:
        st.info(
            f"Dataset loaded from your Downloads folder: "
            f"{DEFAULT_LOCAL_DATASET.name}. You can also upload a different file from the sidebar."
        )

    st.markdown(
        "Use the tabs above to explore analytics, clustering, slotting results, what-if scenarios, "
        "and export Excel reports."
    )

# --- ANALYTICS TAB ---
with tab_analytics:
    st.header("Demand Analysis & Clustering")

    st.subheader("ABC Classification Summary")
    st.dataframe(
        sku_with_abc[["SKU_ID", "Category", "Demand", "ABC_Class"]],
        use_container_width=True,
        height=300
    )

    abc_counts = sku_with_abc["ABC_Class"].value_counts().reindex(["A", "B", "C"]).fillna(0)
    fig_abc = px.bar(
        x=abc_counts.index,
        y=abc_counts.values,
        labels={"x": "ABC Class", "y": "Number of SKUs"},
        title="ABC Class Distribution"
    )
    st.plotly_chart(fig_abc, use_container_width=True)

    st.subheader("SKU Clusters")

    category_filter = st.multiselect(
        "Filter by Category",
        options=sorted(sku_with_clusters["Category"].unique()),
        default=None
    )
    storage_filter = st.multiselect(
        "Filter by Storage Type",
        options=sorted(sku_with_clusters["Storage_Type"].unique()),
        default=None
    )

    df_plot = sku_with_clusters.copy()
    if category_filter:
        df_plot = df_plot[df_plot["Category"].isin(category_filter)]
    if storage_filter:
        df_plot = df_plot[df_plot["Storage_Type"].isin(storage_filter)]

    fig_cluster = px.scatter(
        df_plot,
        x="Volume_m3",
        y="Demand",
        color="Cluster",
        hover_data=["SKU_ID", "Category", "Storage_Type"],
        title="SKU Clusters (Demand vs Volume)"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("SKU Co-occurrence Network")
    if rules_df.empty:
        st.info("No association rules found with current thresholds.")
    else:
        fig_graph = build_cooccurrence_graph(edges_df)
        st.plotly_chart(fig_graph, use_container_width=True)

        st.markdown("Top Association Rules")
        st.dataframe(rules_df.head(20), use_container_width=True, height=300)

# --- SLOTTING TAB ---
with tab_slotting:
    st.header("Slotting Optimization & Zone Utilization (Baseline)")

    col1, col2 = st.columns(2)
    with col1:
        fig_util = plot_zone_utilization(baseline_zone_report)
        st.plotly_chart(fig_util, use_container_width=True)
    with col2:
        fig_sku_zone = plot_skus_per_zone_by_demand_level(
            baseline_assignments, sku_with_abc
        )
        st.plotly_chart(fig_sku_zone, use_container_width=True)

    st.subheader("Baseline SKU → Zone Assignments")
    st.dataframe(
        baseline_assignments.head(200),
        use_container_width=True,
        height=300
    )

    st.subheader("Baseline Zone Utilization Table")
    st.dataframe(
        baseline_zone_report,
        use_container_width=True,
        height=300
    )

# --- SCENARIOS TAB ---
with tab_scenarios:
    st.header("Scenario Comparison")

    scen_names = list(st.session_state.scenarios.keys())
    st.write("Saved scenarios:", ", ".join(scen_names))

    if len(scen_names) < 2:
        st.info("Create at least two scenarios (including Baseline) to compare them.")
    else:
        sel_scenarios = st.multiselect(
            "Select scenarios to compare",
            options=scen_names,
            default=scen_names[:2]
        )
        if sel_scenarios:
            rows = []
            for name in sel_scenarios:
                m = st.session_state.scenarios[name]["metrics"]
                rows.append({
                    "Scenario": name,
                    "Total_Demand_Weighted_Distance": m["Total_Demand_Weighted_Distance"],
                    "Avg_Zone_Utilization": m["Avg_Zone_Utilization"]
                })
            scen_df = pd.DataFrame(rows)

            col1, col2 = st.columns(2)
            with col1:
                fig_dwd = px.bar(
                    scen_df,
                    x="Scenario",
                    y="Total_Demand_Weighted_Distance",
                    title="Total Demand-Weighted Distance by Scenario"
                )
                st.plotly_chart(fig_dwd, use_container_width=True)
            with col2:
                fig_util_comp = px.bar(
                    scen_df,
                    x="Scenario",
                    y="Avg_Zone_Utilization",
                    title="Average Zone Utilization by Scenario (%)"
                )
                st.plotly_chart(fig_util_comp, use_container_width=True)

            st.subheader("Scenario Metrics Table")
            st.dataframe(scen_df, use_container_width=True)

# --- EXPORT TAB ---
with tab_export:
    st.header("Download Results")

    selected_scenario = st.selectbox(
        "Select scenario to export",
        options=list(st.session_state.scenarios.keys()),
        index=0
    )

    scen_data = st.session_state.scenarios[selected_scenario]
    assignments = scen_data["assignments"]
    zones = scen_data["zones"]
    sku_scen = scen_data["sku"]
    metrics = scen_data["metrics"]

    st.write(f"Scenario **{selected_scenario}** metrics:")
    st.json(metrics)

    summary_df = pd.DataFrame([
        {
            "Scenario": selected_scenario,
            "Total_Demand_Weighted_Distance": metrics["Total_Demand_Weighted_Distance"],
            "Avg_Zone_Utilization": metrics["Avg_Zone_Utilization"]
        }
    ])

    cat_kpis = (sku_scen
                .groupby("Category")
                .agg(
                    Total_Demand=("Demand", "sum"),
                    Avg_Demand=("Demand", "mean"),
                    Count_SKUs=("SKU_ID", "nunique")
                )
                .reset_index())

    excel_bytes = to_excel_bytes({
        "SKU_Assignments": assignments,
        "Zone_Utilization": zones,
        "Summary_Report": summary_df,
        "Category_Level_KPIs": cat_kpis
    })

    st.download_button(
        label="Download Excel Report",
        data=excel_bytes,
        file_name=f"Warehouse_Report_{selected_scenario}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
