import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Census Income Intelligence Platform',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #2d3561 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    .main { background-color: #f8f9fc; }

    .page-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a1f36; margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 1rem; color: #6b7280; margin-bottom: 24px;
    }
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #1a1f36;
        border-left: 4px solid #4f6ef7; padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #4f6ef7; margin-bottom: 12px;
    }
    .metric-card-value { font-size: 1.9rem; font-weight: 700; color: #1a1f36; }
    .metric-card-label { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }

    .result-over {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-radius: 16px; padding: 28px; text-align: center;
        border: 2px solid #10b981;
    }
    .result-under {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-radius: 16px; padding: 28px; text-align: center;
        border: 2px solid #ef4444;
    }
    .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }
    .result-sub   { font-size: 0.9rem; color: #374151; }

    .seg-badge-high  { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .seg-badge-above { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .seg-badge-non   { background:#f3f4f6; color:#6b7280; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }


    .conf-bar-bg  { background:#e5e7eb; border-radius:8px; height:12px; margin-top:6px; }
    .conf-bar-fill { border-radius:8px; height:12px; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

    .stButton > button {
        background: linear-gradient(135deg, #4f6ef7, #7c3aed);
        color: white; border: none; border-radius: 10px;
        padding: 14px 0; font-size: 1rem; font-weight: 600;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }
    hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ── load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, 'data')
    clf      = joblib.load(os.path.join(data_dir, 'xgb_model_calibrated.pkl'))
    scaler   = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    X_cols   = pd.read_csv(os.path.join(data_dir, 'X_preprocessed.csv'),
                            nrows=0).columns.tolist()
    return clf, scaler, X_cols

clf, scaler_clf, X_cols = load_models()

# ── constants ─────────────────────────────────────────────────────────────────
COLORS = ['#4f6ef7', '#f87171', '#34d399', '#fbbf24', '#a78bfa']

SEGMENT_NAMES  = {
    0: 'Working Adults — Lower Education',
    1: 'Children & Dependents',
    2: 'Working Adults — Some College',
    3: 'High Earner Investors',
    4: 'Retired / Elderly'
}
SEGMENT_INCOME = {0: 9.3,  1: 0.0,  2: 11.4, 3: 32.7, 4: 1.1}
SEGMENT_SIZE   = {0: 19.7, 1: 27.6, 2: 25.6, 3: 3.7,  4: 23.4}
SEGMENT_AGE    = {0: 39,   1: 8,    2: 38,   3: 48,   4: 55}

SEGMENT_PROFILES = {
    0: dict(sex='Female', marital='Married', occ='Administrative / Clerical',
            edu='High School Graduate', cap=0,   div=15, type='ABOVE AVERAGE TARGET',
            products='Mid-range premium, home improvement, family products',
            channels='Email, social media, online retail',
            message='Value, reliability, aspiration', color='#4f6ef7'),
    1: dict(sex='~50/50', marital='Never married', occ='Not working',
            edu='Children', cap=0, div=0, type='NON-TARGET',
            products='—', channels='—',
            message='Minimal marketing spend recommended', color='#f87171'),
    2: dict(sex='Male', marital='Married', occ='Professional Specialty',
            edu='Some College', cap=0, div=15, type='ABOVE AVERAGE TARGET',
            products='Mid-range premium, home improvement, family products',
            channels='Email, social media, online retail',
            message='Value, reliability, aspiration', color='#34d399'),
    3: dict(sex='Male', marital='Married', occ='Diverse / Mixed',
            edu='Some College+', cap=100, div=28, type='HIGH VALUE TARGET',
            products='Premium goods, financial services, investment products',
            channels='Direct mail, email, premium digital',
            message='Quality, exclusivity, wealth management', color='#fbbf24'),
    4: dict(sex='Female', marital='Married', occ='Not working',
            edu='High School Graduate', cap=0, div=12, type='NON-TARGET',
            products='—', channels='—',
            message='Minimal marketing spend recommended', color='#a78bfa'),
}

EDUCATION_OPTS = [
    'Children', 'Less than 1st grade', '1st 2nd 3rd or 4th grade',
    '5th or 6th grade', '7th and 8th grade', '9th grade', '10th grade',
    '11th grade', '12th grade no diploma', 'High school graduate',
    'Some college but no degree', 'Associates degree-occup /vocational',
    'Associates degree-academic program', 'Bachelors degree(BA AB BS)',
    'Masters degree(MA MS MEng MEd MSW MBA)', 'Doctorate degree(PhD EdD)',
    'Prof school degree (MD DDS DVM LLB JD)'
]
OCCUPATION_OPTS = [
    'Professional specialty', 'Executive admin and managerial', 'Sales',
    'Adm support including clerical', 'Craft repair',
    'Machine operators assmblrs and inspectors',
    'Transportation and material moving', 'Handlers equip cleaners etc',
    'Service', 'Farming forestry and fishing', 'Tech support',
    'Protective services', 'Private household services',
    'Armed Forces', 'Not in universe'
]
INDUSTRY_OPTS = [
    'Finance insurance and real estate', 'Education', 'Medical except hospital',
    'Hospital', 'Retail trade', 'Manufacturing-durable goods',
    'Manufacturing-nondurable goods', 'Public administration', 'Transportation',
    'Business and repair services', 'Personal services without private HH',
    'Entertainment', 'Construction', 'Communications', 'Agriculture',
    'Mining', 'Utilities and sanitary services', 'Wholesale trade',
    'Not in universe or children'
]
WORKER_CLASS_OPTS = [
    'Private', 'Self-employed-incorporated', 'Self-employed-not incorporated',
    'Local government', 'State government', 'Federal government',
    'Without pay', 'Never worked', 'Not in universe'
]
MARITAL_OPTS = [
    'Married-civilian spouse present', 'Married-AF spouse present',
    'Married-spouse absent', 'Divorced', 'Separated', 'Widowed', 'Never married'
]
TAX_FILER_OPTS = [
    'Joint both under 65', 'Joint both 65+', 'Joint one under 65 & one 65+',
    'Single', 'Nonfiler'
]
EMPLOYMENT_OPTS = [
    'Full-time schedules', 'Part-time for non-economic reasons usually full-time',
    'Part-time for economic reasons usually full-time',
    'Part-time for non-economic reasons usually part-time',
    'Unemployed full-time', 'Unemployed part-time',
    'Not in labor force', 'Children or Armed Forces'
]
CITIZENSHIP_OPTS = [
    'Native- Born in the United States',
    'Native- Born in Puerto Rico or U S Outlying',
    'Native- Born abroad of American Parent(s)',
    'Foreign born- U S citizen by naturalization',
    'Foreign born- Not a citizen of U S'
]

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('## 📊 Census Income\n### Intelligence Platform')
    st.markdown('---')
    page = st.radio('', ['🎯  Income Predictor', '👥  Customer Segments'],
                    label_visibility='collapsed')
    st.markdown('---')
    st.markdown('''
    <div style="font-size:0.8rem;opacity:0.75;line-height:1.9">
    <b>Model</b>: XGBoost + Platt Scaling<br>
    <b>AUC-ROC</b>: 0.9546<br>
    <b>CV Mean AUC</b>: 0.9519<br>
    <b>ROI Lift</b>: 10.5x<br>
    <b>Segments</b>: 5 (KMeans)<br>
    <b>Silhouette</b>: 0.2717<br>
    <b>Dataset</b>: 199,523 individuals
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INCOME PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if '🎯' in page:
    st.markdown('<p class="page-title">🎯 Income Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Enter an individual\'s details to predict '
        'their income group and marketing potential.</p>', unsafe_allow_html=True)

    with st.form('prediction_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
            age         = st.slider('Age', 0, 90, 38)
            sex         = st.selectbox('Sex', ['Male', 'Female'])
            education   = st.selectbox('Education Level', EDUCATION_OPTS,
                                        index=EDUCATION_OPTS.index('Bachelors degree(BA AB BS)'))
            marital     = st.selectbox('Marital Status', MARITAL_OPTS)
            citizenship = st.selectbox('Citizenship', CITIZENSHIP_OPTS)

        with col2:
            st.markdown('<div class="section-header">Employment</div>', unsafe_allow_html=True)
            worker_class = st.selectbox('Class of Worker', WORKER_CLASS_OPTS)
            occupation   = st.selectbox('Occupation', OCCUPATION_OPTS)
            industry     = st.selectbox('Industry', INDUSTRY_OPTS)
            employment   = st.selectbox('Employment Status', EMPLOYMENT_OPTS)
            weeks_worked = st.slider('Weeks Worked per Year', 0, 52, 52)
            num_persons  = st.slider('Employer Size', 0, 6, 4,
                          help='0=Not working, 1=<10 employees, 2=10-24, 3=25-99, 4=100-249, 5=250-999, 6=1000+')

        with col3:
            st.markdown('<div class="section-header">Financial</div>', unsafe_allow_html=True)
            tax_filer      = st.selectbox('Tax Filer Status', TAX_FILER_OPTS)
            capital_gains  = st.number_input('Capital Gains ($)', 0, 99998, 0)
            dividends      = st.number_input('Dividend Income ($)', 0, 99998, 0)
            capital_losses = st.number_input('Capital Losses ($)', 0, 4608, 0)
            has_cap        = capital_gains > 0
            has_div        = dividends > 0

        st.markdown('<br>', unsafe_allow_html=True)
        submitted = st.form_submit_button('Run Prediction', width="stretch")

    if submitted:
        person = {
            'age': age, 'sex': sex, 'education': education,
            'marital stat': marital, 'citizenship': citizenship,
            'class of worker': worker_class,
            'major occupation code': occupation,
            'major industry code': industry,
            'full or part time employment stat': employment,
            'weeks worked in year': weeks_worked,
            'num persons worked for employer': num_persons,
            'tax filer stat': tax_filer,
            'capital gains': capital_gains if has_cap else 0,
            'dividends from stocks': dividends if has_div else 0,
            'capital losses': capital_losses,
            'wage per hour': 0,
            'has_capital_gains': int(has_cap),
            'has_dividends': int(has_div),
        }

        df_p = pd.DataFrame([person])
        enc  = pd.get_dummies(df_p)
        enc.columns = (enc.columns
                        .str.replace('[', '_', regex=False)
                        .str.replace(']', '_', regex=False)
                        .str.replace('<', '_', regex=False))
        enc    = enc.reindex(columns=X_cols, fill_value=0)
        scaled = scaler_clf.transform(enc)
        prob   = clf.predict_proba(scaled)[0][1]
        pred   = int(prob >= 0.5)
        pct    = prob * 100

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Prediction Results</div>',
                    unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            html = ('result-over' if pred == 1 else 'result-under')
            icon = '✅ Over $50K' if pred == 1 else '❌ Under $50K'
            sub  = 'High income predicted' if pred == 1 else 'Standard income predicted'
            st.markdown(f'<div class="{html}"><div class="result-title">{icon}</div>'
                        f'<div class="result-sub">{sub}</div></div>',
                        unsafe_allow_html=True)

        with r2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-value">{pct:.1f}%</div>
                <div class="metric-card-label">Probability of Earning Over $50K</div>
            </div>''', unsafe_allow_html=True)

        with r3:
            if prob >= 0.7:
                conf, icon2, col2 = 'High',   '🟢', '#065f46'
            elif prob >= 0.4:
                conf, icon2, col2 = 'Medium', '🟡', '#92400e'
            else:
                conf, icon2, col2 = 'Low',    '🔴', '#991b1b'
            st.markdown(f'''
            <div class="metric-card" style="border-left-color:{col2}">
                <div class="metric-card-value" style="color:{col2}">{icon2} {conf}</div>
                <div class="metric-card-label">Model Confidence Level</div>
            </div>''', unsafe_allow_html=True)

        with r4:
            bar_col = '#10b981' if prob >= 0.5 else '#ef4444'
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-card-label" style="margin-bottom:8px">Probability Score</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill"
                         style="width:{pct:.1f}%;background:{bar_col}"></div>
                </div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:4px">
                    {pct:.1f}% / 100%</div>
            </div>''', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Marketing Campaign Threshold Guide</div>',
                    unsafe_allow_html=True)
        st.caption('Use this table to decide whether to include this person '
                   'based on your campaign strategy.')

        rows = []
        for t, strategy, note in [
            (0.2, 'Broad Reach',  'High volume, lower precision'),
            (0.3, 'Broad Reach',  'Good recall of high earners'),
            (0.4, 'Balanced',     'Balanced precision and recall'),
            (0.5, 'Balanced',     'Default model threshold'),
            (0.6, 'Precision',    'Higher precision, fewer targets'),
            (0.7, 'Precision',    'Highly targeted — 94.1% precision'),
        ]:
            rows.append({'Threshold': t, 'Strategy': strategy, 'Note': note,
                         'Decision': '✅ Target' if prob >= t else '❌ Skip'})

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

        if prob >= 0.7:
            summary_color = '#d1fae5'
            summary_border = '#10b981'
            summary_icon = '🏆'
            summary_text = f'''
            <b>Strong Target.</b> This individual has a <b>{pct:.1f}% probability</b> of earning over $50K -
            well above the population average of 6.2%. We recommend including them in
            <b>all campaigns including premium and precision targeting</b>.
            Based on their profile, they are likely a high-value customer worth significant marketing investment.
            '''
        elif prob >= 0.5:
            summary_color = '#dbeafe'
            summary_border = '#3b82f6'
            summary_icon = '✅'
            summary_text = f'''
            <b>Good Target.</b> This individual has a <b>{pct:.1f}% probability</b> of earning over $50K -
            above the population average of 6.2%. We recommend including them in
            <b>standard and broad reach campaigns</b>.
            They show promising signals but may not justify premium campaign spend.
            '''
        elif prob >= 0.3:
            summary_color = '#fef3c7'
            summary_border = '#f59e0b'
            summary_icon = '⚠️'
            summary_text = f'''
            <b>Moderate Candidate.</b> This individual has a <b>{pct:.1f}% probability</b> of earning over $50K.
            They are slightly above the population average of 6.2% but below the recommended targeting threshold.
            Consider including them only in <b>high-volume broad reach campaigns</b> where cost per contact is low.
            Do not allocate premium marketing budget to this individual.
            '''
        else:
            summary_color = '#fee2e2'
            summary_border = '#ef4444'
            summary_icon = '❌'
            summary_text = f'''
            <b>Not Recommended.</b> This individual has only a <b>{pct:.1f}% probability</b> of earning over $50K -
            below the population average of 6.2%. Marketing spend on this individual is unlikely to generate
            meaningful return. <b>We recommend excluding them from targeted campaigns.</b>
            '''

        st.markdown(f'''
        <div style="background:{summary_color};border-left:5px solid {summary_border};
                    border-radius:12px;padding:20px 24px;margin-top:8px">
            <div style="font-size:1.1rem;margin-bottom:8px">{summary_icon} <b>What This Means for Your Campaign</b></div>
            <div style="font-size:0.95rem;color:#1a1f36;line-height:1.8">{summary_text}</div>
            <div style="font-size:0.82rem;color:#6b7280;margin-top:12px">
            Model: XGBoost with Platt Scaling &nbsp;|&nbsp;
            AUC-ROC: 0.9546 &nbsp;|&nbsp;
            10.5x ROI improvement over random targeting
            </div>
        </div>
        ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif '👥' in page:
    st.markdown('<p class="page-title">👥 Customer Segments</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Five customer segments discovered via KMeans '
        'clustering on 14 demographic and employment features. '
        'Silhouette score: 0.2717.</p>', unsafe_allow_html=True)

    # KPI row
    cols = st.columns(5)
    for i, col in enumerate(cols):
        p = SEGMENT_PROFILES[i]
        with col:
            st.markdown(f'''
            <div class="metric-card" style="border-left-color:{p["color"]};text-align:center">
                <div style="font-size:0.78rem;font-weight:600;color:#6b7280;margin-bottom:4px">
                    Segment {i}</div>
                <div style="font-size:1.7rem;font-weight:700;color:{p["color"]}">
                    {SEGMENT_INCOME[i]}%</div>
                <div style="font-size:0.72rem;color:#6b7280">earn over $50K</div>
                <div style="font-size:0.8rem;font-weight:600;color:#1a1f36;margin-top:6px">
                    {SEGMENT_SIZE[i]}% of pop.</div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    ch1, ch2 = st.columns([3, 2])

    with ch1:
        st.markdown('<div class="section-header">Income Rate by Segment</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#f8f9fc')
        ax.set_facecolor('#f8f9fc')
        short = [f'Seg {i}\n{SEGMENT_NAMES[i].split("—")[0].split("&")[0].strip()[:14]}'
                 for i in range(5)]
        bars = ax.bar(short, [SEGMENT_INCOME[i] for i in range(5)],
                      color=COLORS, width=0.55, zorder=3)
        ax.axhline(6.2, color='#94a3b8', linestyle='--', linewidth=1.5,
                   label='Overall avg (6.2%)', zorder=2)
        ax.set_ylabel('% Earning Over $50K', fontsize=11, color='#6b7280')
        ax.set_ylim(0, 40)
        ax.legend(fontsize=10)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_color('#e5e7eb')
        ax.tick_params(colors='#6b7280')
        ax.grid(axis='y', color='#e5e7eb', linewidth=0.8, zorder=0)
        for bar, i in zip(bars, range(5)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    SEGMENT_INCOME[i] + 0.5,
                    f'{SEGMENT_INCOME[i]}%',
                    ha='center', fontsize=10.5, fontweight='600', color='#1a1f36')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with ch2:
        st.markdown('<div class="section-header">Population Split</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#f8f9fc')
        wedges, _, autotexts = ax2.pie(
            [SEGMENT_SIZE[i] for i in range(5)],
            colors=COLORS, autopct='%1.1f%%',
            startangle=90, pctdistance=0.78,
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight('600')
            at.set_color('white')
        ax2.legend(
            [f'Seg {i}: {SEGMENT_NAMES[i].split("—")[0].strip()[:16]}' for i in range(5)],
            loc='lower center', bbox_to_anchor=(0.5, -0.22),
            fontsize=7.5, ncol=2
        )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Detailed Segment Profiles</div>',
                unsafe_allow_html=True)

    for i in range(5):
        p = SEGMENT_PROFILES[i]
        if p['type'] == 'HIGH VALUE TARGET':
            badge = f'<span class="seg-badge-high">🏆 {p["type"]}</span>'
        elif p['type'] == 'ABOVE AVERAGE TARGET':
            badge = f'<span class="seg-badge-above">⭐ {p["type"]}</span>'
        else:
            badge = f'<span class="seg-badge-non">⚠️ {p["type"]}</span>'

        with st.expander(
            f'Segment {i}  -  {SEGMENT_NAMES[i]}  '
            f'|  {SEGMENT_SIZE[i]}% of population  '
            f'|  {SEGMENT_INCOME[i]}% earn >$50K',
            expanded=(i == 3)
        ):
            st.markdown(f'<div style="margin-bottom:12px">{badge}</div>',
                        unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)

            with d1:
                st.markdown('**📋 Demographics**')
                st.markdown(f'''
| Field | Value |
|---|---|
| Avg Age | {SEGMENT_AGE[i]} years |
| Population | {SEGMENT_SIZE[i]}% |
| Sex | {p["sex"]} |
| Marital Status | {p["marital"]} |
| Education | {p["edu"]} |
''')
            with d2:
                st.markdown('**💼 Employment & Finance**')
                st.markdown(f'''
| Field | Value |
|---|---|
| Occupation | {p["occ"]} |
| Capital Gains | {p["cap"]}% of segment |
| Dividends | {p["div"]}% of segment |
| Income >$50K | {SEGMENT_INCOME[i]}% |
''')
            with d3:
                st.markdown('**🎯 Marketing Strategy**')
                if p['type'] != 'NON-TARGET':
                    st.markdown(f'''
| Field | Value |
|---|---|
| Products | {p["products"]} |
| Channels | {p["channels"]} |
| Message | {p["message"]} |
''')
                else:
                    st.warning(p['message'])
                    st.caption('Budget is better allocated to Segments 0, 2, and 3.')



