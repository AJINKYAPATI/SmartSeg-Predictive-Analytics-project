import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import pickle
import streamlit as st
import os
from datetime import datetime
import squarify
import base64

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SegmentIQ · Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --bg:          #F5F5F7;
    --surface:     #FFFFFF;
    --accent:      #0071E3;
    --accent-soft: #EBF3FD;
    --text:        #1D1D1F;
    --text-2:      #6E6E73;
    --border:      #E5E5E7;
    --shadow-sm:   0 2px 8px rgba(0,0,0,.06);
    --shadow-md:   0 8px 32px rgba(0,0,0,.09);
    --shadow-lg:   0 20px 60px rgba(0,0,0,.12);
    --radius:      16px;
    --radius-sm:   10px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1320px;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    box-shadow: var(--shadow-md);
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stDateInput"] > div > div > input {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus,
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,.12) !important;
    outline: none !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 980px !important;
    padding: .55rem 1.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    letter-spacing: .01em !important;
    transition: background .2s, transform .15s, box-shadow .2s !important;
    box-shadow: 0 2px 12px rgba(0,113,227,.28) !important;
}
.stButton > button:hover {
    background: #0062c9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,113,227,.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stFileUploader"] > div {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--accent-soft) !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"] > div:hover { border-color: var(--accent) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ─── REUSABLE UI HELPERS ─────────────────────────────────────────────────────────

def section_title(text, sub=None):
    sub_html = f'<p style="margin:4px 0 0;font-size:.95rem;color:var(--text-2);font-weight:400;">{sub}</p>' if sub else ""
    st.markdown(f"""
    <div style="margin:2rem 0 1rem;">
        <h2 style="font-family:'DM Serif Display',serif;font-size:1.7rem;
                   font-weight:400;color:var(--text);margin:0;letter-spacing:-.02em;">{text}</h2>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def metric_card(label, value, sub=None, accent=False):
    bg   = "var(--accent)"  if accent else "var(--surface)"
    col  = "#fff"            if accent else "var(--text)"
    col2 = "rgba(255,255,255,.7)" if accent else "var(--text-2)"
    bdr  = "transparent"     if accent else "var(--border)"
    shd  = "0 8px 28px rgba(0,113,227,.30)" if accent else "var(--shadow-sm)"
    sub_html = f'<p style="margin:6px 0 0;font-size:.8rem;color:{col2};font-weight:500;">{sub}</p>' if sub else ""
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {bdr};border-radius:var(--radius);
                padding:1.4rem 1.6rem;box-shadow:{shd};">
        <p style="margin:0;font-size:.78rem;font-weight:600;letter-spacing:.08em;
                  text-transform:uppercase;color:{col2};">{label}</p>
        <p style="margin:6px 0 0;font-size:2rem;font-weight:700;
                  color:{col};letter-spacing:-.03em;line-height:1;">{value}</p>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def divider():
    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1.5rem 0;">', unsafe_allow_html=True)


def info_box(text, icon="ℹ️"):
    st.markdown(f"""
    <div style="background:var(--accent-soft);border:1px solid #d0e8fc;
                border-radius:var(--radius-sm);padding:1rem 1.2rem;
                display:flex;gap:.75rem;align-items:flex-start;margin:.75rem 0;">
        <span style="font-size:1.1rem;">{icon}</span>
        <span style="font-size:.9rem;color:var(--text);line-height:1.5;">{text}</span>
    </div>""", unsafe_allow_html=True)


def page_header(icon, title, subtitle, dark=False):
    bg  = "linear-gradient(135deg,#1D1D1F 0%,#2D2D2F 100%)" if dark else "var(--surface)"
    tc  = "#fff"            if dark else "var(--text)"
    tc2 = "rgba(255,255,255,.55)" if dark else "var(--text-2)"
    ic_bg = "rgba(0,113,227,.25)" if dark else "var(--accent-soft)"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid var(--border);border-radius:var(--radius);
                padding:1.8rem 2.2rem;margin-bottom:1.5rem;box-shadow:var(--shadow-sm);">
        <div style="display:flex;align-items:center;gap:1rem;">
            <div style="width:44px;height:44px;background:{ic_bg};border-radius:12px;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <span style="font-size:1.3rem;">{icon}</span>
            </div>
            <div>
                <h2 style="margin:0;font-size:1.3rem;font-weight:700;color:{tc};letter-spacing:-.02em;">{title}</h2>
                <p style="margin:2px 0 0;font-size:.85rem;color:{tc2};">{subtitle}</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def plotly_base_layout():
    return dict(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='DM Sans', size=13, color='#1D1D1F'),
        title_font=dict(family='DM Sans', size=15, color='#1D1D1F'),
        margin=dict(l=10, r=10, t=55, b=10),
    )


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.8rem 1.6rem 1rem;border-bottom:1px solid var(--border);margin-bottom:.5rem;">
        <div style="display:flex;align-items:center;gap:.7rem;">
            <div style="width:36px;height:36px;background:var(--accent);border-radius:10px;
                        display:flex;align-items:center;justify-content:center;">
                <span style="font-size:1.1rem;color:#fff;">◈</span>
            </div>
            <div>
                <p style="margin:0;font-size:1rem;font-weight:700;color:var(--text);letter-spacing:-.02em;">SegmentIQ</p>
                <p style="margin:0;font-size:.72rem;color:var(--text-2);font-weight:400;">Customer Intelligence</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="padding:.4rem 1.6rem;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text-2);margin:0;">Navigation</p>', unsafe_allow_html=True)

    menu  = ["Business Understanding","Data Understanding","Data preparation","Modeling & Evaluation","Predict"]
    icons = ["◉","◎","◑","◈","◆"]
    choice = st.selectbox("", menu, label_visibility="collapsed")

    for i, item in enumerate(menu):
        active = item == choice
        bg  = "#EBF3FD"   if active else "transparent"
        col = "#0071E3"   if active else "#6E6E73"
        fw  = "600"       if active else "400"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.6rem;padding:.55rem 1.2rem;
                    margin:.15rem .6rem;border-radius:10px;background:{bg};">
            <span style="font-size:.8rem;color:{col};">{icons[i]}</span>
            <span style="font-size:.88rem;font-weight:{fw};color:{col};">{item}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:1.5rem;left:0;width:260px;padding:0 1.2rem;">
        <div style="background:#EBF3FD;border:1px solid #d0e8fc;border-radius:12px;padding:.85rem 1rem;">
            <p style="margin:0;font-size:.78rem;font-weight:600;color:#0071E3;">v2.0 · Powered by KMeans</p>
            <p style="margin:2px 0 0;font-size:.72rem;color:#6E6E73;">RFM · Elbow Method · Clustering</p>
        </div>
    </div>""", unsafe_allow_html=True)


# ─── BACKEND HELPERS (logic unchanged) ──────────────────────────────────────────

def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(
            uploaded_file,
            encoding='utf-8',
            encoding_errors='ignore',
            sep=r'\s+',
            header=None,
            names=['Customer_id', 'day', 'Quantity', 'Sales']
        )
        df.to_csv("CDNOW_master_new.txt", index=False)
        df['day'] = pd.to_datetime(df['day'], format='%Y%m%d')
        st.session_state['df'] = df
        return df
    else:
        info_box("Please upload a data file to proceed.", "📂")
        return None


def csv_download_link(df, csv_file_name, download_link_text):
    csv_data = df.to_csv(index=True)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f"""<a href="data:file/csv;base64,{b64}" download="{csv_file_name}" style="
        display:inline-flex;align-items:center;gap:.5rem;background:var(--accent);color:#fff;
        text-decoration:none;padding:.55rem 1.4rem;border-radius:999px;font-size:.88rem;
        font-weight:600;box-shadow:0 4px 14px rgba(0,113,227,.3);">⬇ {download_link_text}</a>"""
    st.markdown(href, unsafe_allow_html=True)


# ─── SESSION STATE ───────────────────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None


# ════════════════════════════════════════════════════════════════════════════════
# PAGE : BUSINESS UNDERSTANDING
# ════════════════════════════════════════════════════════════════════════════════
if choice == 'Business Understanding':

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0071E3 0%,#0A84FF 60%,#34AADC 100%);
                border-radius:24px;padding:3rem 3.5rem;margin-bottom:2rem;
                position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-60px;width:260px;height:260px;
                    background:rgba(255,255,255,.08);border-radius:50%;"></div>
        <div style="position:absolute;bottom:-80px;right:120px;width:180px;height:180px;
                    background:rgba(255,255,255,.05);border-radius:50%;"></div>
        <p style="margin:0 0 .4rem;font-size:.8rem;font-weight:700;letter-spacing:.12em;
                  text-transform:uppercase;color:rgba(255,255,255,.65);">SegmentIQ Platform</p>
        <h1 style="margin:0 0 .8rem;font-family:'DM Serif Display',serif;font-size:3rem;
                   font-weight:400;color:#fff;line-height:1.1;letter-spacing:-.03em;">
            Customer<br><em>Intelligence Suite</em>
        </h1>
        <p style="margin:0;font-size:1.05rem;color:rgba(255,255,255,.82);max-width:520px;line-height:1.6;">
            Harness machine learning to segment your customers with precision —
            enabling hyper-personalised strategies and measurable growth.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Personalization", "1:1",  "Tailored strategies per segment", accent=True)
    with c2: metric_card("Optimization",   "↑ ROI", "Efficient resource allocation")
    with c3: metric_card("Insight",        "360°",  "Deep customer understanding")
    with c4: metric_card("Engagement",     "+ NPS", "Satisfaction & retention")

    divider()
    section_title("Business Objective", "What problem does this platform solve?")

    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
                    padding:1.8rem 2rem;box-shadow:var(--shadow-sm);">
            <p style="font-size:1rem;line-height:1.8;color:var(--text);margin:0;">
                Customer segmentation is a fundamental task in <strong>marketing and customer
                relationship management</strong>. With the advancements in data analytics and
                machine learning, it is now possible to group customers into distinct segments
                with a high degree of precision — allowing businesses to tailor their marketing
                strategies and offerings to each segment's unique needs and preferences.
            </p>
            <div style="margin-top:1.2rem;padding-top:1.2rem;border-top:1px solid var(--border);">
                <p style="font-size:.85rem;font-weight:600;color:var(--text-2);
                           letter-spacing:.06em;text-transform:uppercase;margin:0 0 .6rem;">
                    Problem Statement
                </p>
                <p style="font-size:.95rem;color:var(--text);margin:0;line-height:1.6;">
                    Utilise machine learning and data analysis techniques in Python to perform
                    customer segmentation using the RFM framework combined with KMeans clustering.
                </p>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        for icon, title, desc in [
            ("◈","Personalization","Tailor every campaign to segment needs"),
            ("◉","Optimization",   "Allocate budgets where they convert best"),
            ("◎","Insight",        "Understand who your best customers are"),
            ("◆","Engagement",     "Increase satisfaction & reduce churn"),
        ]:
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border);
                        border-radius:var(--radius-sm);padding:1rem 1.2rem;margin-bottom:.65rem;
                        box-shadow:var(--shadow-sm);display:flex;align-items:flex-start;gap:.9rem;">
                <span style="font-size:1.1rem;color:var(--accent);margin-top:.05rem;">{icon}</span>
                <div>
                    <p style="margin:0;font-size:.88rem;font-weight:700;color:var(--text);">{title}</p>
                    <p style="margin:2px 0 0;font-size:.8rem;color:var(--text-2);">{desc}</p>
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE : DATA UNDERSTANDING
# ════════════════════════════════════════════════════════════════════════════════
elif choice == 'Data Understanding':

    page_header("📊","Data Understanding","Load your dataset and explore its structure before analysis.")

    sample_files = os.listdir('data')
    data_source  = st.sidebar.radio('Data source', ['Use a sample file', 'Upload a new file'])

    col_src, col_prev = st.columns([1, 2], gap="large")

    with col_src:
        st.markdown("""<div style="background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius);padding:1.4rem 1.6rem;box-shadow:var(--shadow-sm);">
            <p style="margin:0 0 1rem;font-size:.78rem;font-weight:700;letter-spacing:.08em;
                      text-transform:uppercase;color:var(--text-2);">Data Source</p>
        """, unsafe_allow_html=True)

        if data_source == 'Use a sample file':
            selected_file = st.sidebar.selectbox('Choose a sample file', sample_files)
            st.markdown(f"""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
                <span style="font-size:1rem;">📁</span>
                <span style="font-size:.9rem;font-weight:600;color:var(--text);">{selected_file}</span>
            </div>""", unsafe_allow_html=True)
            file_path = os.path.join('data', selected_file)
            st.session_state['uploaded_file'] = open(file_path, 'r')
            load_data(st.session_state['uploaded_file'])
        else:
            st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
            if st.session_state['uploaded_file'] is not None:
                load_data(st.session_state['uploaded_file'])

        st.markdown("</div>", unsafe_allow_html=True)

    with col_prev:
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Rows",    f"{df.shape[0]:,}",  "Total records")
            with c2: metric_card("Columns", str(df.shape[1]),     "Features")
            with c3: metric_card("Timespan",
                                 f"{df['day'].dt.year.min()}–{df['day'].dt.year.max()}",
                                 "Date range", accent=True)
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            st.markdown('<p style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:0 0 .5rem;">Preview — First 5 Rows</p>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
        else:
            info_box("No data loaded yet. Choose a source from the sidebar.", "📂")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE : DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════════
elif choice == 'Data preparation':

    page_header("🔧","Data Preparation","Clean, validate, and transform the dataset for modelling.")

    if st.session_state['df'] is not None:
        df = st.session_state['df']

        section_title("Data Quality Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Missing Values",   str(int(df.isnull().sum().sum())), "Across all columns")
        with c2: metric_card("NA Strings",       str(int((df == 'NA').sum().sum())), "String nulls")
        with c3: metric_card("Duplicates",       str(int(df.duplicated().sum())),    "Duplicate rows")
        with c4: metric_card("Unique Customers", f"{df['Customer_id'].nunique():,}","Distinct IDs", accent=True)

        divider()
        section_title("Cleaning Controls")

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("""<div style="background:var(--surface);border:1px solid var(--border);
                border-radius:var(--radius);padding:1.4rem 1.6rem;box-shadow:var(--shadow-sm);">
                <p style="margin:0 0 .8rem;font-size:.78rem;font-weight:700;letter-spacing:.08em;
                text-transform:uppercase;color:var(--text-2);">Actions</p>""", unsafe_allow_html=True)
            if st.checkbox('Remove duplicate rows'):
                st.session_state['df'].drop_duplicates(inplace=True)
                st.success("✓ Duplicate rows removed.")
            if st.checkbox('Remove rows with NA values'):
                st.session_state['df'].replace('NA', pd.NA, inplace=True)
                st.session_state['df'].dropna(inplace=True)
                st.success("✓ Rows with NA values removed.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("""<div style="background:var(--surface);border:1px solid var(--border);
                border-radius:var(--radius);padding:1.4rem 1.6rem;box-shadow:var(--shadow-sm);">
                <p style="margin:0 0 .8rem;font-size:.78rem;font-weight:700;letter-spacing:.08em;
                text-transform:uppercase;color:var(--text-2);">Unique Value Counts</p>""",
                unsafe_allow_html=True)
            st.dataframe(df.nunique().rename("Unique Count"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        divider()
        section_title("Distribution Analysis", "Histograms for numeric columns")

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in num_cols:
            fig = px.histogram(df, x=col, nbins=50, template="simple_white",
                               color_discrete_sequence=["#0071E3"], title=col)
            fig.update_layout(**plotly_base_layout(),
                              bargap=0.08,
                              xaxis=dict(showgrid=False, linecolor="#E5E5E7"),
                              yaxis=dict(gridcolor="#F5F5F7", linecolor="#E5E5E7"))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        divider()
        section_title("Outlier Analysis", "Box plots for numeric columns")

        for col in num_cols:
            fig = px.box(df, y=col, template="simple_white",
                         color_discrete_sequence=["#0071E3"], title=col)
            fig.update_layout(**plotly_base_layout(),
                              yaxis=dict(gridcolor="#F5F5F7"))
            st.plotly_chart(fig, use_container_width=True)

        divider()
        section_title("Data Transformation")

        st.markdown(f"""
        <div style="background:var(--accent-soft);border:1px solid #d0e8fc;
                    border-radius:var(--radius-sm);padding:1rem 1.4rem;margin-bottom:1rem;">
            <span style="font-size:.9rem;color:var(--accent);font-weight:600;">
                📅 Transactions: {df['day'].min().strftime('%B %d, %Y')} → {df['day'].max().strftime('%B %d, %Y')}
                &nbsp;·&nbsp; {df[df.Customer_id.isnull()].shape[0]:,} transactions without customer ID
                &nbsp;·&nbsp; {df['Customer_id'].nunique():,} unique customers
            </span>
        </div>""", unsafe_allow_html=True)

        user_grouped = df.groupby('Customer_id').agg({'Quantity': 'sum', 'Sales': 'sum'})

        col_t1, col_t2 = st.columns(2, gap="large")
        with col_t1:
            st.markdown('<p style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:0 0 .5rem;">User Grouped Data</p>', unsafe_allow_html=True)
            st.dataframe(user_grouped.head(), use_container_width=True)
        with col_t2:
            df['month'] = df['day'].values.astype('datetime64[M]')
            st.markdown('<p style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:0 0 .5rem;">Data with Month Column</p>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)

        divider()
        section_title("Time Series Trends")

        dfm  = df.groupby('month')['Quantity'].sum().reset_index()
        dfpc = df.groupby('month')['Sales'].sum().reset_index()

        col_ch1, col_ch2 = st.columns(2, gap="large")
        with col_ch1:
            fig1 = px.area(dfm, x='month', y='Quantity', template='simple_white',
                           color_discrete_sequence=['#0071E3'], title='Total Quantity per Month')
            fig1.update_traces(line_width=2.5)
            fig1.update_layout(**plotly_base_layout(),
                               xaxis=dict(showgrid=False, linecolor="#E5E5E7"),
                               yaxis=dict(gridcolor="#F5F5F7"), hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)
        with col_ch2:
            fig2 = px.area(dfpc, x='month', y='Sales', template='simple_white',
                           color_discrete_sequence=['#34C759'], title='Total Sales per Month')
            fig2.update_traces(line_width=2.5)
            fig2.update_layout(**plotly_base_layout(),
                               xaxis=dict(showgrid=False, linecolor="#E5E5E7"),
                               yaxis=dict(gridcolor="#F5F5F7"), hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

        section_title("Sales vs Quantity Scatter")
        col_s1, col_s2 = st.columns(2, gap="large")

        def scatter_fig(data, x, y, title):
            fig = px.scatter(data, x=x, y=y, template="simple_white",
                             color_discrete_sequence=["#0071E3"], title=title, opacity=0.55)
            fig.update_traces(marker_size=5)
            fig.update_layout(**plotly_base_layout(),
                              xaxis=dict(gridcolor="#F5F5F7"),
                              yaxis=dict(gridcolor="#F5F5F7"))
            return fig

        with col_s1:
            st.plotly_chart(scatter_fig(df, 'Sales', 'Quantity', 'Individual Transactions'), use_container_width=True)
        with col_s2:
            st.plotly_chart(scatter_fig(user_grouped.reset_index(), 'Sales', 'Quantity', 'User Grouped Data'), use_container_width=True)

    else:
        info_box("No data available. Please upload a file in the 'Data Understanding' section.", "⚠️")

    divider()
    section_title("User Feedback")
    user_feedback = st.text_area("Share your comments or feedback:", value='',
                                 placeholder="What did you think of this section?", height=100)
    if st.button("Submit Feedback", key="fb_prep"):
        current_time = datetime.now()
        feedback_df  = pd.DataFrame({'Time': [current_time], 'Feedback': [user_feedback]})
        if not os.path.isfile('feedback.csv'):
            feedback_df.to_csv('feedback.csv', index=False)
        else:
            feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
        st.success("✓ Your feedback has been recorded!")

    if os.path.isfile('feedback.csv'):
        all_feedbacks = pd.read_csv('feedback.csv')
        all_feedbacks.sort_values('Time', ascending=False, inplace=True)
        st.markdown('<p style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:1.2rem 0 .5rem;">5 Most Recent Feedbacks</p>', unsafe_allow_html=True)
        st.dataframe(all_feedbacks.head(5), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE : MODELING & EVALUATION
# ════════════════════════════════════════════════════════════════════════════════
elif choice == 'Modeling & Evaluation':

    page_header("🤖","Modeling & Evaluation",
                "KMeans clustering with RFM features · Elbow method for optimal k", dark=True)

    if st.session_state['df'] is not None:
        df = st.session_state['df']

        # RFM computation (logic unchanged)
        recent_date = df['day'].max()
        df_RFM = df.groupby('Customer_id').agg({
            'day':         lambda x: (recent_date - x.max()).days,
            'Customer_id': 'count',
            'Sales':       'sum'
        }).rename(columns={'day': 'Recency', 'Customer_id': 'Frequency', 'Sales': 'Monetary'})

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Avg Recency",   f"{df_RFM['Recency'].mean():.0f}d",  "Days since last purchase")
        with c2: metric_card("Avg Frequency", f"{df_RFM['Frequency'].mean():.1f}", "Orders per customer")
        with c3: metric_card("Avg Monetary",  f"${df_RFM['Monetary'].mean():.0f}", "Lifetime value", accent=True)

        divider()
        section_title("Elbow Method", "Determine the optimal number of clusters")

        sse = {}
        with st.spinner("Computing inertia for k = 1 … 19"):
            for k in range(1, 20):
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(df_RFM)
                sse[k] = km.inertia_

        elbow_df = pd.DataFrame({'k': list(sse.keys()), 'SSE': list(sse.values())})
        fig_elbow = px.line(elbow_df, x='k', y='SSE', markers=True, template='simple_white',
                            color_discrete_sequence=['#0071E3'],
                            title='Sum of Squared Distances vs Number of Clusters')
        fig_elbow.update_traces(marker_size=7, line_width=2.5)
        fig_elbow.update_layout(**plotly_base_layout(),
                                xaxis=dict(title='Number of Clusters (k)', gridcolor='#F5F5F7', dtick=1),
                                yaxis=dict(title='SSE', gridcolor='#F5F5F7'),
                                hovermode='x unified')
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Cluster selector (logic unchanged)
        n_clusters = st.sidebar.number_input(
            'Select number of clusters k (2–20):',
            min_value=2, max_value=20, value=3, step=1, key="cluster_value"
        )
        info_box(f"<strong>{n_clusters} clusters</strong> selected — KMeans will partition customers into {n_clusters} distinct segments.", "◈")

        # KMeans fit (logic unchanged)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(df_RFM)

        df_sub            = df_RFM.copy()
        df_sub['Cluster'] = model.labels_

        cluster_stats = df_sub.groupby('Cluster').agg({
            'Recency':   'mean',
            'Frequency': 'mean',
            'Monetary':  ['mean', 'count']
        }).round(2)
        cluster_stats.columns = ['RecencyMean','FrequencyMean','MonetaryMean','Count']
        cluster_stats['Percent'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(2)
        cluster_stats.reset_index(inplace=True)
        cluster_stats['Cluster'] = 'Cluster ' + cluster_stats['Cluster'].astype('str')

        divider()
        section_title("Cluster Statistics")
        st.dataframe(cluster_stats, use_container_width=True)

        divider()
        section_title("Visual Analytics")

        # Scatter
        fig_scatter = px.scatter(
            cluster_stats, x='RecencyMean', y='MonetaryMean',
            size='FrequencyMean', color='Cluster', log_x=True, size_max=60,
            template='simple_white',
            title='Cluster Scatter — Recency vs Monetary (size = Frequency)',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_scatter.update_layout(**plotly_base_layout())
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Treemap (matplotlib — logic unchanged)
        colors_dict = {0:'#0071E3',1:'#FF3B30',2:'#34C759',3:'#FF9500',4:'#AF52DE'}
        fig_treemap, ax_treemap = plt.subplots(facecolor='#FFFFFF')
        fig_treemap.set_size_inches(14, 9)
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        squarify.plot(
            sizes=cluster_stats['Count'],
            label=[
                f'Cluster {i}\n{row.RecencyMean:.0f}d recency\n{row.FrequencyMean:.1f} orders\n${row.MonetaryMean:.0f} LTV\n{row.Count} customers ({row.Percent}%)'
                for i, row in cluster_stats.iterrows()
            ],
            color=[colors_dict.get(c, '#0071E3') for c in cluster_stats.index],
            alpha=0.88,
            text_kwargs={'fontsize':12,'fontweight':'bold','color':'white'},
            pad=True, ax=ax_treemap
        )
        ax_treemap.set_title("Customer Segmentation — Treemap",
                             fontsize=22, fontweight='bold', pad=20, color='#1D1D1F')
        ax_treemap.axis('off')
        fig_treemap.tight_layout()
        st.pyplot(fig_treemap)

        # 3D scatter
        fig_3d = px.scatter_3d(
            cluster_stats, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
            color='Cluster', size='Count',
            labels={'RecencyMean':'Recency','FrequencyMean':'Frequency','MonetaryMean':'Monetary'},
            template='simple_white', title='3D Cluster View',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_3d.update_layout(paper_bgcolor='white',
                             font=dict(family='DM Sans',size=12,color='#1D1D1F'),
                             title_font=dict(family='DM Sans',size=15,color='#1D1D1F'),
                             margin=dict(l=0,r=0,t=55,b=0),
                             scene=dict(
                                 xaxis=dict(backgroundcolor='#F5F5F7',gridcolor='#E5E5E7'),
                                 yaxis=dict(backgroundcolor='#F5F5F7',gridcolor='#E5E5E7'),
                                 zaxis=dict(backgroundcolor='#F5F5F7',gridcolor='#E5E5E7')))
        st.plotly_chart(fig_3d, use_container_width=True)

        divider()
        col_exp, _ = st.columns([1,3])
        with col_exp:
            if st.button('⬇  Export Model', key='export_btn'):
                with open('kmeans_model.pkl', 'wb') as f:
                    pickle.dump((model, cluster_stats), f)
                st.session_state.model_exported = True
                st.success("✓ Model (kmeans_model.pkl) exported successfully!")

        divider()
        section_title("User Feedback")
        user_feedback = st.text_area("Share your comments or feedback:", value='',
                                     placeholder="Any thoughts on the model results?", height=100)
        if st.button("Submit Feedback", key="fb_model"):
            current_time = datetime.now()
            feedback_df  = pd.DataFrame({'Time':[current_time],'Feedback':[user_feedback]})
            if not os.path.isfile('feedback.csv'):
                feedback_df.to_csv('feedback.csv', index=False)
            else:
                feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
            st.success("✓ Your feedback has been recorded!")

        if os.path.isfile('feedback.csv'):
            all_feedbacks = pd.read_csv('feedback.csv')
            all_feedbacks.sort_values('Time', ascending=False, inplace=True)
            st.markdown('<p style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:1rem 0 .5rem;">5 Most Recent Feedbacks</p>', unsafe_allow_html=True)
            st.dataframe(all_feedbacks.head(5), use_container_width=True)

    else:
        info_box("No data available. Please upload a file in the 'Data Understanding' section.", "⚠️")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE : PREDICT
# ════════════════════════════════════════════════════════════════════════════════
elif choice == 'Predict':

    st.markdown("""
    <div style="background:linear-gradient(135deg,#034AA6 0%,#0071E3 100%);
                border-radius:var(--radius);padding:1.8rem 2.2rem;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:1rem;">
            <div style="width:44px;height:44px;background:rgba(255,255,255,.15);border-radius:12px;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <span style="font-size:1.3rem;">🔮</span>
            </div>
            <div>
                <h2 style="margin:0;font-size:1.3rem;font-weight:700;color:#fff;letter-spacing:-.02em;">
                    Predict Customer Cluster</h2>
                <p style="margin:2px 0 0;font-size:.85rem;color:rgba(255,255,255,.65);">
                    Add new customer records and predict their segment using the trained model.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if 'model_exported' in st.session_state and st.session_state.model_exported:
        with open('kmeans_model.pkl', 'rb') as f:
            model, cluster_stats = pickle.load(f)

        section_title("Trained Model — Cluster Reference")
        st.dataframe(cluster_stats, use_container_width=True)

        divider()
        section_title("Add New Customer")

        st.markdown("""<div style="background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius);padding:1.6rem 1.8rem;box-shadow:var(--shadow-sm);">""",
            unsafe_allow_html=True)

        col_i1, col_i2, col_i3, col_i4 = st.columns(4, gap="medium")
        with col_i1: customer_name = st.text_input('Customer Name:', placeholder='e.g. Alice')
        with col_i2: recent_date   = st.date_input('Most Recent Purchase Date:')
        with col_i3: quantity      = st.number_input('Quantity:', min_value=0)
        with col_i4: monetary      = st.number_input('Amount ($):', min_value=0.0)

        st.markdown("</div>", unsafe_allow_html=True)

        if 'df_new' not in st.session_state:
            st.session_state['df_new'] = pd.DataFrame(columns=['Customer_id','day','Quantity','Sales'])

        col_btn1, _ = st.columns([1, 7])
        with col_btn1:
            if st.button("＋  Add Customer"):
                new_data = pd.DataFrame({
                    'Customer_id':[customer_name],'day':[recent_date],
                    'Quantity':[quantity],'Sales':[monetary]
                })
                if 'df_new' not in st.session_state:
                    st.session_state['df_new'] = new_data
                else:
                    st.session_state['df_new'] = pd.concat(
                        [st.session_state['df_new'], new_data], ignore_index=True)

        if not st.session_state['df_new'].empty:
            st.markdown('<p style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-2);margin:1rem 0 .5rem;">Added Records</p>', unsafe_allow_html=True)
            st.dataframe(st.session_state['df_new'], use_container_width=True)

        divider()
        col_pred, _ = st.columns([1, 6])
        with col_pred:
            predict_btn = st.button("🔮  Predict Segments")

        if predict_btn:
            # Logic unchanged
            recent_date = pd.Timestamp.now().date()
            df_RFM = st.session_state['df_new'].groupby('Customer_id').agg({
                'day':         lambda x: (recent_date - x.max()).days,
                'Customer_id': 'count',
                'Sales':       'sum'
            }).rename(columns={'day':'Recency','Customer_id':'Frequency','Sales':'Monetary'})

            cluster_pred      = model.predict(df_RFM)
            df_RFM['Cluster'] = cluster_pred

            section_title("Prediction Results")
            cluster_colors = ['#0071E3','#FF3B30','#34C759','#FF9500','#AF52DE']

            for idx, row in df_RFM.iterrows():
                cl  = int(row['Cluster'])
                clr = cluster_colors[cl % len(cluster_colors)]
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);
                            border-radius:var(--radius);padding:1.2rem 1.6rem;margin-bottom:.7rem;
                            box-shadow:var(--shadow-sm);display:flex;align-items:center;
                            gap:2rem;flex-wrap:wrap;">
                    <div>
                        <p style="margin:0;font-size:.7rem;font-weight:700;letter-spacing:.1em;
                                  text-transform:uppercase;color:var(--text-2);">Customer</p>
                        <p style="margin:2px 0 0;font-size:1rem;font-weight:700;color:var(--text);">{idx}</p>
                    </div>
                    <div>
                        <p style="margin:0;font-size:.7rem;font-weight:700;letter-spacing:.1em;
                                  text-transform:uppercase;color:var(--text-2);">Recency</p>
                        <p style="margin:2px 0 0;font-size:1rem;font-weight:700;color:var(--text);">{int(row['Recency'])}d</p>
                    </div>
                    <div>
                        <p style="margin:0;font-size:.7rem;font-weight:700;letter-spacing:.1em;
                                  text-transform:uppercase;color:var(--text-2);">Frequency</p>
                        <p style="margin:2px 0 0;font-size:1rem;font-weight:700;color:var(--text);">{int(row['Frequency'])}</p>
                    </div>
                    <div>
                        <p style="margin:0;font-size:.7rem;font-weight:700;letter-spacing:.1em;
                                  text-transform:uppercase;color:var(--text-2);">Monetary</p>
                        <p style="margin:2px 0 0;font-size:1rem;font-weight:700;color:var(--text);">${row['Monetary']:.2f}</p>
                    </div>
                    <div style="margin-left:auto;">
                        <span style="background:{clr}18;color:{clr};font-size:.85rem;font-weight:700;
                                     padding:.35rem 1rem;border-radius:999px;">Cluster {cl}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.dataframe(df_RFM, use_container_width=True)
            st.markdown("<div style='margin-top:.8rem;'>", unsafe_allow_html=True)
            csv_download_link(df_RFM, 'RFM_prediction_results.csv', 'Download Prediction Results')
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        info_box("You must <strong>export the model</strong> in the Modeling & Evaluation section before making predictions.", "🔒")

    divider()
    section_title("User Feedback")
    user_feedback = st.text_area("Share your comments or feedback:", value='',
                                 placeholder="How accurate were the predictions?", height=100)
    if st.button("Submit Feedback", key="fb_predict"):
        current_time = datetime.now()
        feedback_df  = pd.DataFrame({'Time':[current_time],'Feedback':[user_feedback]})
        if not os.path.isfile('feedback.csv'):
            feedback_df.to_csv('feedback.csv', index=False)
        else:
            feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
        st.success("✓ Your feedback has been recorded!")
