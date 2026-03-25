import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from utils import (load_data, encode_for_ml, get_ml_features,
                   FEATURE_LABELS, GOAL_LABELS, PAIN_LABELS, INV_LABELS,
                   INCOME_ORDER, AGE_ORDER, RISK_ORDER, WTP_ORDER,
                   BINARY_COLS, ORDINAL_MAPS, CLUSTER_PALETTE, PERSONA_COLORS)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InvestIQ — Platform Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  .metric-card { background: #f8f9fa; border-radius: 12px; padding: 1rem 1.25rem;
                 border: 0.5px solid #e0e0e0; }
  .section-title { font-size: 1.1rem; font-weight: 600; color: #1a1a2e;
                   border-bottom: 2px solid #534AB7; padding-bottom: 6px; margin-bottom: 1rem; }
  .insight-box { background: #EEEDFE; border-left: 4px solid #534AB7;
                 border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
                 margin: 0.5rem 0; font-size: 0.9rem; color: #26215C; }
  .persona-card { border-radius: 12px; padding: 1rem; border: 1px solid #e0e0e0; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 InvestIQ Intelligence")
    st.markdown("*Data-driven platform for investor insights*")
    st.divider()

    uploaded = st.file_uploader("Upload survey CSV", type=['csv'],
        help="Upload a new survey CSV in the same format to analyse or predict.")
    st.divider()

    st.markdown("**Navigation**")
    page = st.radio("", [
        "🏠 Overview",
        "📋 Descriptive Analysis",
        "🔍 Diagnostic Analysis",
        "👥 Customer Segmentation",
        "🎯 Classification Model",
        "🔗 Association Rule Mining",
        "📈 Regression Analysis",
        "💡 Prescriptive Intelligence",
        "🚀 New Customer Predictor",
    ], label_visibility="collapsed")

    st.divider()
    st.caption("Built for InvestIQ · Powered by ML")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data(file=None):
    return load_data(file)

df = get_data(uploaded)

# ── Helper: colour map ────────────────────────────────────────────────────────
def adoption_color(val):
    return PERSONA_COLORS.get(val, '#888')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("InvestIQ Platform Intelligence Dashboard")
    st.markdown("*Founder's command centre — turning survey data into strategic decisions*")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Respondents", f"{len(df):,}")
    c2.metric("Likely Adopters", f"{(df['Q25_adoption_3class']=='Likely').sum():,}",
              f"{(df['Q25_adoption_3class']=='Likely').mean()*100:.1f}%")
    c3.metric("Avg App Comfort", f"{df['Q20_app_comfort'].mean():.2f}/5")
    c4.metric("Avg NPS Score", f"{df['Q24_nps'].mean():.1f}/10")
    c5.metric("Paid WTP (>₹99)", f"{(df['Q23_wtp_numeric']>=2).sum():,}",
              f"{(df['Q23_wtp_numeric']>=2).mean()*100:.1f}%")

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-title">Adoption Intent</div>', unsafe_allow_html=True)
        vc = df['Q25_adoption_3class'].value_counts()
        fig = px.pie(values=vc.values, names=vc.index,
                     color=vc.index,
                     color_discrete_map=PERSONA_COLORS,
                     hole=0.55)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Willingness to Pay</div>', unsafe_allow_html=True)
        wtp_vc = df['Q23_wtp'].value_counts().reindex(WTP_ORDER).fillna(0)
        fig2 = px.bar(x=wtp_vc.index, y=wtp_vc.values,
                      color=wtp_vc.values,
                      color_continuous_scale='Purples',
                      labels={'x':'WTP Tier','y':'Respondents'})
        fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown('<div class="section-title">Top Desired Features</div>', unsafe_allow_html=True)
        feat_sums = df[[c for c in FEATURE_LABELS]].sum().sort_values(ascending=True)
        feat_sums.index = [FEATURE_LABELS[i] for i in feat_sums.index]
        fig3 = px.bar(x=feat_sums.values, y=feat_sums.index, orientation='h',
                      color=feat_sums.values, color_continuous_scale='Teal')
        fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                           coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-title">Key Founder Insights</div>', unsafe_allow_html=True)
    ia, ib = st.columns(2)
    with ia:
        likely_pct = (df['Q25_adoption_3class']=='Likely').mean()*100
        paid_pct   = (df['Q23_wtp_numeric']>=2).mean()*100
        metro_likely = df[df['Q2_region']=='Metro']['Q25_adoption_3class'].value_counts(normalize=True).get('Likely',0)*100
        st.markdown(f'<div class="insight-box">📌 <b>{likely_pct:.1f}%</b> of surveyed Indians are likely platform adopters — a significant addressable base given India\'s ~80M fintech-ready population.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">💰 <b>{paid_pct:.1f}%</b> are willing to pay ₹99+/month — validating a freemium-to-premium monetisation strategy.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">🏙️ Metro respondents show <b>{metro_likely:.1f}%</b> likely adoption — your primary beachhead market.</div>', unsafe_allow_html=True)
    with ib:
        top_feat = df[[c for c in FEATURE_LABELS]].sum().idxmax()
        top_pain = df[[c for c in PAIN_LABELS]].sum().idxmax()
        avg_comfort = df['Q20_app_comfort'].mean()
        st.markdown(f'<div class="insight-box">⭐ Most demanded feature: <b>{FEATURE_LABELS[top_feat]}</b> — build this into your MVP core.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">🚧 Biggest customer pain: <b>{PAIN_LABELS[top_pain]}</b> — your platform must directly address this in onboarding.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">📱 Average app comfort score: <b>{avg_comfort:.2f}/5</b> — UX simplicity is non-negotiable for mass adoption.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Descriptive Analysis":
    st.title("📋 Descriptive Analysis")
    st.markdown("*Who are our respondents and what does the data look like?*")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics","Investment Behaviour","Features & Goals","Trust & Budget"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='Q1_age', category_orders={'Q1_age':AGE_ORDER},
                               color='Q1_age', color_discrete_sequence=px.colors.qualitative.Set2,
                               title="Age distribution")
            fig.update_layout(showlegend=False, xaxis_title="Age group", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.histogram(df, x='Q5_income', category_orders={'Q5_income':INCOME_ORDER},
                                color='Q25_adoption_3class',
                                color_discrete_map=PERSONA_COLORS,
                                barmode='group', title="Income vs adoption intent")
            fig2.update_layout(xaxis_title="Annual income", yaxis_title="Count", legend_title="Adoption")
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            reg_vc = df['Q2_region'].value_counts()
            fig3 = px.pie(values=reg_vc.values, names=reg_vc.index, title="Region distribution",
                          color_discrete_sequence=CLUSTER_PALETTE, hole=0.4)
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.histogram(df, x='Q4_occupation', color='Q2_region',
                                barmode='group', title="Occupation by region",
                                color_discrete_sequence=CLUSTER_PALETTE)
            fig4.update_layout(xaxis_tickangle=-30, xaxis_title='', yaxis_title='Count')
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            inv_sums = df[[c for c in INV_LABELS]].sum().sort_values(ascending=False)
            inv_sums.index = [INV_LABELS[i] for i in inv_sums.index]
            fig = px.bar(x=inv_sums.index, y=inv_sums.values,
                         color=inv_sums.values, color_continuous_scale='Blues',
                         title="Current investment products used")
            fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False,
                              xaxis_title='', yaxis_title='Respondents')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.histogram(df, x='Q9_risk_appetite',
                                category_orders={'Q9_risk_appetite': RISK_ORDER},
                                color='Q1_age',
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                barmode='group', title="Risk appetite by age group")
            fig2.update_layout(xaxis_title='Risk appetite', yaxis_title='Count',
                               legend_title='Age', xaxis_tickangle=-20)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x='Q7_monthly_investment',
                            color='Q5_income',
                            color_discrete_sequence=CLUSTER_PALETTE,
                            barmode='stack', title="Monthly investment amount by income bracket")
        fig3.update_layout(xaxis_title='Monthly investment', yaxis_title='Count', legend_title='Income')
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        c1,c2 = st.columns(2)
        with c1:
            feat_sums = df[[c for c in FEATURE_LABELS]].sum().sort_values(ascending=False)
            feat_sums.index = [FEATURE_LABELS[i] for i in feat_sums.index]
            fig = px.bar(x=feat_sums.values, y=feat_sums.index, orientation='h',
                         color=feat_sums.values, color_continuous_scale='Purples',
                         title="Feature demand ranking")
            fig.update_layout(coloraxis_showscale=False, yaxis_title='', xaxis_title='Respondents')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            goal_sums = df[[c for c in GOAL_LABELS]].sum().sort_values(ascending=False)
            goal_sums.index = [GOAL_LABELS[i] for i in goal_sums.index]
            fig2 = px.bar(x=goal_sums.values, y=goal_sums.index, orientation='h',
                          color=goal_sums.values, color_continuous_scale='Teal',
                          title="Financial goal ranking")
            fig2.update_layout(coloraxis_showscale=False, yaxis_title='', xaxis_title='Respondents')
            st.plotly_chart(fig2, use_container_width=True)

        pain_sums = df[[c for c in PAIN_LABELS]].sum().sort_values(ascending=False)
        pain_sums.index = [PAIN_LABELS[i] for i in pain_sums.index]
        fig3 = px.bar(x=pain_sums.index, y=pain_sums.values,
                      color=pain_sums.values, color_continuous_scale='Reds',
                      title="Customer pain points — what's blocking them")
        fig3.update_layout(coloraxis_showscale=False, xaxis_title='', yaxis_title='Respondents',
                           xaxis_tickangle=-25)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        c1,c2 = st.columns(2)
        with c1:
            wtp_vc = df['Q23_wtp'].value_counts().reindex(WTP_ORDER).fillna(0)
            fig = px.bar(x=wtp_vc.index, y=wtp_vc.values,
                         color=wtp_vc.values, color_continuous_scale='Greens',
                         title="Willingness to pay distribution")
            fig.update_layout(coloraxis_showscale=False, xaxis_title='', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.box(df, x='Q25_adoption_3class', y='Q24_nps',
                          color='Q25_adoption_3class',
                          color_discrete_map=PERSONA_COLORS,
                          title="NPS score distribution by adoption intent")
            fig2.update_layout(showlegend=False, xaxis_title='Adoption intent', yaxis_title='NPS score')
            st.plotly_chart(fig2, use_container_width=True)

        trust_sums = df[['trust_sebi','trust_security','trust_brand','trust_reviews',
                         'trust_fees','trust_trial','trust_media']].sum().sort_values(ascending=False)
        trust_labels = {'trust_sebi':'SEBI/RBI reg.','trust_security':'Bank-grade security',
                        'trust_brand':'Brand partnership','trust_reviews':'User reviews',
                        'trust_fees':'Fee transparency','trust_trial':'Free trial','trust_media':'Media coverage'}
        trust_sums.index = [trust_labels[i] for i in trust_sums.index]
        fig3 = px.bar(x=trust_sums.index, y=trust_sums.values,
                      color=trust_sums.values, color_continuous_scale='Blues',
                      title="What builds trust in a new platform")
        fig3.update_layout(coloraxis_showscale=False, xaxis_title='', yaxis_title='Respondents')
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analysis":
    st.title("🔍 Diagnostic Analysis")
    st.markdown("*Why do certain customer groups behave differently?*")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap","Crosstab Explorer","Behavioural Insights"])

    with tab1:
        st.markdown("#### Correlation matrix — numerical & ordinal features")
        dfe = encode_for_ml(df)
        num_cols = [c+'_enc' for c in ['Q1_age','Q5_income','Q3_education',
                                        'Q7_monthly_investment','Q9_time_horizon',
                                        'Q9_risk_appetite','Q23_wtp','Q25_adoption_3class']
                    if c+'_enc' in dfe.columns]
        num_cols += ['Q19_tax_importance','Q20_app_comfort','Q24_nps','Q23_wtp_numeric']
        num_cols = [c for c in num_cols if c in dfe.columns]

        corr = dfe[num_cols].corr().round(2)
        clean_labels = [c.replace('_enc','').replace('Q','').replace('_',' ') for c in num_cols]

        fig = px.imshow(corr.values,
                        x=clean_labels, y=clean_labels,
                        color_continuous_scale='RdBu', zmin=-1, zmax=1,
                        text_auto=True, aspect='auto',
                        title="Pearson correlation — ordinal features")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">🔍 Strong positive correlations between income, monthly investment, willingness to pay and adoption score confirm that financial capacity is the primary adoption driver. App comfort correlates strongly with adoption independently of income — meaning UX investment has ROI across all income segments.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Crosstab explorer — compare any two dimensions")
        cat_cols = ['Q1_age','Q2_region','Q3_education','Q4_occupation','Q5_income',
                    'Q7_monthly_investment','Q9_risk_appetite','Q14_autonomy',
                    'Q23_wtp','Q25_adoption_3class','Q11_dropout']
        c1,c2 = st.columns(2)
        x_var = c1.selectbox("X axis (rows)", cat_cols, index=0)
        y_var = c2.selectbox("Y axis (colour)", cat_cols, index=10)

        ct = pd.crosstab(df[x_var], df[y_var], normalize='index').round(3)*100
        fig2 = px.imshow(ct.values, x=ct.columns.tolist(), y=ct.index.tolist(),
                         color_continuous_scale='Purples', text_auto='.1f',
                         labels={'color':'%'}, title=f"{x_var} vs {y_var} (row %)")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Raw crosstab counts")
        ct2 = pd.crosstab(df[x_var], df[y_var], margins=True)
        st.dataframe(ct2, use_container_width=True)

    with tab3:
        st.markdown("#### Loss aversion vs risk appetite")
        la_risk = pd.crosstab(df['Q13_loss_aversion'], df['Q9_risk_appetite'], normalize='index')*100
        fig = px.imshow(la_risk.values, x=la_risk.columns.tolist(), y=la_risk.index.tolist(),
                        color_continuous_scale='RdYlGn', text_auto='.1f',
                        title="Loss aversion scenario choice vs self-reported risk appetite (row %)")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Decision autonomy vs platform adoption")
        c1,c2 = st.columns(2)
        with c1:
            aut_adopt = df.groupby('Q14_autonomy')['Q25_adoption_3class'].value_counts(normalize=True).unstack().fillna(0)*100
            fig2 = px.bar(aut_adopt, barmode='stack',
                          color_discrete_map=PERSONA_COLORS,
                          title="Autonomy style vs adoption intent (%)")
            fig2.update_layout(xaxis_title='', yaxis_title='%', legend_title='Adoption')
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            drop_adopt = df.groupby('Q11_dropout')['Q25_adoption_3class'].value_counts(normalize=True).unstack().fillna(0)*100
            fig3 = px.bar(drop_adopt, barmode='stack',
                          color_discrete_map=PERSONA_COLORS,
                          title="Dropout behaviour vs adoption intent (%)")
            fig3.update_layout(xaxis_title='', yaxis_title='%', legend_title='Adoption',
                               xaxis_tickangle=-20)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="insight-box">💡 Respondents who previously dropped off citing "No trust" show the highest "Unlikely" adoption rate — meaning trust-building features (SEBI badge, transparent fees) are not just nice-to-have but conversion-critical for this segment. Conversely, DIY autonomy users who dropped off for "Too complicated" are recoverable with better UX.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("*K-Means clustering to discover natural investor personas*")
    st.divider()

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X, feat_cols = get_ml_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tab1, tab2, tab3 = st.tabs(["Optimal K","Persona Profiles","Cluster Deep Dive"])

    with tab1:
        st.markdown("#### Finding optimal number of clusters")
        k_range = range(2, 9)
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, name='Inertia (elbow)',
                                  line=dict(color='#534AB7', width=2), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=list(k_range), y=sil_scores, name='Silhouette score',
                                  line=dict(color='#1D9E75', width=2), mode='lines+markers'),
                      secondary_y=True)
        fig.update_layout(title='Elbow method + Silhouette score', xaxis_title='Number of clusters (K)',
                          height=350)
        fig.update_yaxes(title_text="Inertia", secondary_y=False)
        fig.update_yaxes(title_text="Silhouette score", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">📊 Silhouette score peaks guide the optimal K. We recommend K=5 which balances interpretability with statistical cohesion — giving 5 actionable investor personas.</div>', unsafe_allow_html=True)

    with tab2:
        K = st.slider("Select number of clusters", 3, 7, 5)
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        df['cluster'] = km.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        df['pca1'], df['pca2'] = X_pca[:,0], X_pca[:,1]

        fig = px.scatter(df, x='pca1', y='pca2', color=df['cluster'].astype(str),
                         symbol='Q25_adoption_3class',
                         color_discrete_sequence=CLUSTER_PALETTE,
                         title=f"Customer clusters in PCA space (K={K})",
                         labels={'color':'Cluster','symbol':'Adoption'},
                         opacity=0.65, hover_data=['Q1_age','Q5_income','Q2_region'])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Cluster profiles — key characteristics")
        cluster_summary = df.groupby('cluster').agg(
            Size=('cluster','count'),
            Avg_App_Comfort=('Q20_app_comfort','mean'),
            Avg_NPS=('Q24_nps','mean'),
            Likely_Pct=('Q25_adoption_3class', lambda x: (x=='Likely').mean()*100),
            Avg_WTP=('Q23_wtp_numeric','mean'),
            Avg_Income_Idx=('Q5_income', lambda x: x.map({v:i for i,v in enumerate(['<3L','3-6L','6-12L','12-25L','25-50L','>50L'])}).mean()),
        ).round(2)
        st.dataframe(cluster_summary.style.background_gradient(cmap='Purples', subset=['Likely_Pct'])
                     .background_gradient(cmap='Greens', subset=['Avg_WTP'])
                     .format({'Likely_Pct':'{:.1f}%','Avg_App_Comfort':'{:.2f}',
                              'Avg_NPS':'{:.1f}','Avg_WTP':'{:.2f}','Avg_Income_Idx':'{:.2f}'}),
                     use_container_width=True)

        persona_names = {
            0: "🚀 Digital Natives", 1: "💼 Cautious Savers", 2: "🏆 Wealth Builders",
            3: "🌱 Aspiring Beginners", 4: "👨‍💼 Delegation Seekers",
            5: "🔬 DIY Researchers", 6: "🏘️ Conservative Traditionalists"
        }
        st.markdown("#### Persona cards")
        cols = st.columns(min(K, 3))
        for i in range(K):
            c_data = df[df['cluster']==i]
            with cols[i % 3]:
                name = persona_names.get(i, f"Cluster {i}")
                likely_pct = (c_data['Q25_adoption_3class']=='Likely').mean()*100
                color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                st.markdown(f"""
                <div style="border:1px solid {color};border-radius:12px;padding:1rem;margin-bottom:1rem">
                  <div style="color:{color};font-weight:600;font-size:1rem">{name}</div>
                  <div style="font-size:0.8rem;color:#666;margin-top:4px">n = {len(c_data)}</div>
                  <hr style="margin:8px 0;border-color:{color}33">
                  <div style="font-size:0.85rem">
                    📱 App comfort: <b>{c_data['Q20_app_comfort'].mean():.1f}/5</b><br>
                    📈 Adoption likely: <b>{likely_pct:.0f}%</b><br>
                    💰 Avg WTP tier: <b>{c_data['Q23_wtp_numeric'].mean():.1f}/5</b><br>
                    ⭐ Avg NPS: <b>{c_data['Q24_nps'].mean():.1f}/10</b><br>
                    🏙️ Top region: <b>{c_data['Q2_region'].mode()[0]}</b>
                  </div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        K2 = st.slider("Clusters for deep dive", 3, 7, 5, key='k2')
        km2 = KMeans(n_clusters=K2, random_state=42, n_init=10)
        df['cluster2'] = km2.fit_predict(X_scaled)

        sel_cluster = st.selectbox("Select cluster to analyse", list(range(K2)))
        c_df = df[df['cluster2']==sel_cluster]

        c1,c2,c3 = st.columns(3)
        c1.metric("Cluster size", len(c_df))
        c2.metric("% of total", f"{len(c_df)/len(df)*100:.1f}%")
        c3.metric("Adoption rate", f"{(c_df['Q25_adoption_3class']=='Likely').mean()*100:.1f}%")

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(c_df, x='Q1_age', category_orders={'Q1_age':AGE_ORDER},
                               title=f"Cluster {sel_cluster} — Age distribution",
                               color_discrete_sequence=[CLUSTER_PALETTE[sel_cluster % len(CLUSTER_PALETTE)]])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(c_df, x='Q5_income', category_orders={'Q5_income':['<3L','3-6L','6-12L','12-25L','25-50L','>50L']},
                                title=f"Cluster {sel_cluster} — Income distribution",
                                color_discrete_sequence=[CLUSTER_PALETTE[sel_cluster % len(CLUSTER_PALETTE)]])
            st.plotly_chart(fig2, use_container_width=True)

        feat_comp = pd.DataFrame({
            'Feature': [FEATURE_LABELS[c] for c in FEATURE_LABELS],
            'Cluster': [c_df[c].mean() for c in FEATURE_LABELS],
            'Overall': [df[c].mean() for c in FEATURE_LABELS],
        })
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name=f'Cluster {sel_cluster}', x=feat_comp['Feature'],
                               y=feat_comp['Cluster'], marker_color=CLUSTER_PALETTE[sel_cluster % len(CLUSTER_PALETTE)]))
        fig3.add_trace(go.Bar(name='Overall avg', x=feat_comp['Feature'],
                               y=feat_comp['Overall'], marker_color='#ccc'))
        fig3.update_layout(barmode='group', title='Feature preference: cluster vs overall',
                           xaxis_tickangle=-30, height=380)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Classification Model":
    st.title("🎯 Classification Model")
    st.markdown("*Predicting platform adoption intent — Likely / Neutral / Unlikely*")
    st.divider()

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, roc_curve)

    X, feat_cols = get_ml_features(df)
    y = df['Q25_adoption_3class']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    tab1, tab2, tab3 = st.tabs(["Model Performance","Feature Importance","ROC Curves"])

    with tab1:
        model_choice = st.selectbox("Select model", ["Random Forest","Logistic Regression","Gradient Boosting"])

        @st.cache_resource
        def train_model(name):
            if name == "Random Forest":
                m = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            elif name == "Logistic Regression":
                m = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            else:
                m = GradientBoostingClassifier(n_estimators=100, random_state=42)
            m.fit(X_train_s, y_train)
            return m

        model = train_model(model_choice)
        y_pred = model.predict(X_test_s)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5)

        c1,c2,c3,c4 = st.columns(4)
        from sklearn.metrics import accuracy_score, f1_score
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro')*100:.1f}%")
        c3.metric("CV mean (5-fold)", f"{cv_scores.mean()*100:.1f}%")
        c4.metric("CV std", f"±{cv_scores.std()*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        labels = le.classes_
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale='Purples',
                        title=f"Confusion matrix — {model_choice}",
                        labels={'x':'Predicted','y':'Actual'})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Classification report")
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['f1-score']),
                     use_container_width=True)

    with tab2:
        rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False).head(20)

        clean_names = []
        for f in importances.index:
            n = f.replace('_enc','').replace('Q','Q').replace('_',' ')
            clean_names.append(n)

        fig = px.bar(x=importances.values, y=clean_names, orientation='h',
                     color=importances.values, color_continuous_scale='Purples',
                     title="Top 20 features — Random Forest importance")
        fig.update_layout(coloraxis_showscale=False, yaxis_title='', height=520,
                          yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">🎯 The top predictors of adoption intent are app comfort, willingness to pay, and data sharing consent — all behavioural/attitudinal signals, not just demographics. This means your product experience and trust-building are more important than who you market to.</div>', unsafe_allow_html=True)

    with tab3:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=[0,1,2])
        y_prob = model.predict_proba(X_test_s)
        fig = go.Figure()
        colors_roc = ['#534AB7','#1D9E75','#D85A30']
        for i, cls in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_prob[:,i])
            auc = roc_auc_score(y_test_bin[:,i], y_prob[:,i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{cls} (AUC={auc:.3f})',
                                      line=dict(color=colors_roc[i], width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='grey'),
                                  name='Random baseline', showlegend=True))
        fig.update_layout(title='ROC curves — one vs rest',
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          height=420, legend=dict(x=0.6, y=0.1))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rule Mining":
    st.title("🔗 Association Rule Mining")
    st.markdown("*Finding hidden co-occurrence patterns using Apriori algorithm*")
    st.divider()

    from mlxtend.frequent_patterns import apriori, association_rules

    tab1, tab2, tab3 = st.tabs(["Investment Patterns","Feature Combos","Goal & Pain Associations"])

    def run_arm(cols, min_sup, min_conf, min_lift, label_map=None):
        basket = df[cols].astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        if len(freq) == 0:
            return pd.DataFrame()
        rules = association_rules(freq, metric='lift', min_threshold=min_lift)
        rules = rules[rules['confidence'] >= min_conf].sort_values('lift', ascending=False)
        if label_map:
            rules['antecedents_str'] = rules['antecedents'].apply(
                lambda x: ', '.join([label_map.get(i,i) for i in x]))
            rules['consequents_str'] = rules['consequents'].apply(
                lambda x: ', '.join([label_map.get(i,i) for i in x]))
        else:
            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(x))
            rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(x))
        return rules

    with tab1:
        st.markdown("#### Investment product co-occurrence rules")
        c1,c2,c3 = st.columns(3)
        sup = c1.slider("Min support", 0.05, 0.5, 0.15, 0.05, key='s1')
        conf = c2.slider("Min confidence", 0.3, 0.9, 0.5, 0.05, key='c1')
        lift = c3.slider("Min lift", 1.0, 3.0, 1.2, 0.1, key='l1')

        inv_cols = list(INV_LABELS.keys())
        rules = run_arm(inv_cols, sup, conf, lift, INV_LABELS)
        if not rules.empty:
            st.markdown(f"**{len(rules)} rules found**")
            display_cols = ['antecedents_str','consequents_str','support','confidence','lift']
            st.dataframe(rules[display_cols].head(20)
                        .rename(columns={'antecedents_str':'If investor uses',
                                         'consequents_str':'...they also use',
                                         'support':'Support','confidence':'Confidence','lift':'Lift'})
                        .style.background_gradient(cmap='Purples', subset=['Lift'])
                        .format({'Support':'{:.3f}','Confidence':'{:.3f}','Lift':'{:.3f}'}),
                        use_container_width=True)

            top = rules.head(15)
            fig = px.scatter(top, x='support', y='confidence', size='lift',
                             color='lift', color_continuous_scale='Purples',
                             hover_data=['antecedents_str','consequents_str'],
                             title='Top rules — support vs confidence (size = lift)',
                             labels={'antecedents_str':'Antecedent','consequents_str':'Consequent'})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No rules found. Try reducing min support or confidence thresholds.")

    with tab2:
        st.markdown("#### Feature preference co-occurrence")
        c1,c2,c3 = st.columns(3)
        sup2  = c1.slider("Min support", 0.05, 0.5, 0.20, 0.05, key='s2')
        conf2 = c2.slider("Min confidence", 0.3, 0.9, 0.55, 0.05, key='c2')
        lift2 = c3.slider("Min lift", 1.0, 3.0, 1.1, 0.1, key='l2')

        feat_cols_arm = list(FEATURE_LABELS.keys())
        rules2 = run_arm(feat_cols_arm, sup2, conf2, lift2, FEATURE_LABELS)
        if not rules2.empty:
            st.markdown(f"**{len(rules2)} rules found**")
            st.dataframe(rules2[['antecedents_str','consequents_str','support','confidence','lift']].head(20)
                        .rename(columns={'antecedents_str':'Features selected together',
                                         'consequents_str':'Predict also wanting',
                                         'support':'Support','confidence':'Confidence','lift':'Lift'})
                        .style.background_gradient(cmap='Greens', subset=['Lift'])
                        .format({'Support':'{:.3f}','Confidence':'{:.3f}','Lift':'{:.3f}'}),
                        use_container_width=True)
            st.markdown('<div class="insight-box">🔗 High-lift feature associations reveal natural product bundles. If two features always appear together, bundle them in the same subscription tier — users who want one will expect the other.</div>', unsafe_allow_html=True)
        else:
            st.warning("No rules found. Try reducing thresholds.")

    with tab3:
        st.markdown("#### Goal + pain point associations")
        c1,c2,c3 = st.columns(3)
        sup3  = c1.slider("Min support", 0.05, 0.5, 0.10, 0.05, key='s3')
        conf3 = c2.slider("Min confidence", 0.3, 0.9, 0.45, 0.05, key='c3')
        lift3 = c3.slider("Min lift", 1.0, 3.0, 1.1, 0.1, key='l3')

        gp_cols = list(GOAL_LABELS.keys()) + list(PAIN_LABELS.keys())
        gp_labels = {**GOAL_LABELS, **PAIN_LABELS}
        rules3 = run_arm(gp_cols, sup3, conf3, lift3, gp_labels)
        if not rules3.empty:
            st.dataframe(rules3[['antecedents_str','consequents_str','support','confidence','lift']].head(20)
                        .rename(columns={'antecedents_str':'Goal/Pain','consequents_str':'Associated with',
                                         'support':'Support','confidence':'Confidence','lift':'Lift'})
                        .style.background_gradient(cmap='Oranges', subset=['Lift'])
                        .format({'Support':'{:.3f}','Confidence':'{:.3f}','Lift':'{:.3f}'}),
                        use_container_width=True)
        else:
            st.warning("No rules found. Try reducing thresholds.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression Analysis":
    st.title("📈 Regression Analysis")
    st.markdown("*Predicting willingness to pay — what drives a customer up the value ladder?*")
    st.divider()

    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    X, feat_cols = get_ml_features(df)
    y_reg = df['Q23_wtp_numeric'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    tab1, tab2, tab3 = st.tabs(["Model Comparison","Coefficient Analysis","Predicted vs Actual"])

    with tab1:
        models_reg = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.05),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        results = []
        trained_models = {}
        for name, m in models_reg.items():
            m.fit(X_train_s, y_train)
            yp = m.predict(X_test_s)
            results.append({
                'Model': name,
                'R² Score': round(r2_score(y_test, yp), 4),
                'MAE': round(mean_absolute_error(y_test, yp), 4),
                'RMSE': round(np.sqrt(mean_squared_error(y_test, yp)), 4),
            })
            trained_models[name] = m

        results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
        st.dataframe(results_df.style.background_gradient(cmap='Greens', subset=['R² Score'])
                     .background_gradient(cmap='Reds_r', subset=['MAE','RMSE']),
                     use_container_width=True)

        fig = px.bar(results_df, x='Model', y='R² Score',
                     color='R² Score', color_continuous_scale='Greens',
                     title='Model R² comparison — willingness to pay prediction')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">📈 Random Forest and Gradient Boosting typically achieve highest R² for this type of survey data because the relationships between features and WTP are non-linear. Linear models serve as interpretable baselines.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Ridge regression coefficients — feature impact on WTP")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        coefs = pd.Series(ridge.coef_, index=feat_cols).sort_values()

        clean = [c.replace('_enc','').replace('_',' ') for c in coefs.index]
        colors = ['#D85A30' if v < 0 else '#1D9E75' for v in coefs.values]

        fig = go.Figure(go.Bar(
            x=coefs.values, y=clean, orientation='h',
            marker_color=colors
        ))
        fig.update_layout(title='Ridge coefficients — green = increases WTP, red = decreases WTP',
                          xaxis_title='Coefficient', yaxis_title='', height=600,
                          yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Random Forest feature importance for WTP")
        rf_reg = trained_models['Random Forest']
        imp = pd.Series(rf_reg.feature_importances_, index=feat_cols).sort_values(ascending=False).head(20)
        clean2 = [c.replace('_enc','').replace('_',' ') for c in imp.index]
        fig2 = px.bar(x=imp.values, y=clean2, orientation='h',
                      color=imp.values, color_continuous_scale='Greens',
                      title="Top 20 predictors of willingness to pay")
        fig2.update_layout(coloraxis_showscale=False, yaxis=dict(autorange='reversed'), height=460)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        model_sel = st.selectbox("Choose model", list(models_reg.keys()))
        m_sel = trained_models[model_sel]
        y_pred_plot = m_sel.predict(X_test_s)

        fig = px.scatter(x=y_test, y=y_pred_plot,
                         labels={'x':'Actual WTP tier','y':'Predicted WTP tier'},
                         title=f'Predicted vs Actual WTP — {model_sel}',
                         opacity=0.5, color_discrete_sequence=['#534AB7'])
        fig.add_shape(type='line', x0=0, x1=5, y0=0, y1=5,
                      line=dict(color='red', dash='dash'))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        residuals = y_test - y_pred_plot
        fig2 = px.histogram(x=residuals, title='Residual distribution',
                            color_discrete_sequence=['#534AB7'],
                            labels={'x':'Residual (actual - predicted)'})
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PRESCRIPTIVE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Prescriptive Intelligence":
    st.title("💡 Prescriptive Intelligence")
    st.markdown("*From data to strategy — actionable founder recommendations*")
    st.divider()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X, feat_cols = get_ml_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)

    tab1, tab2, tab3 = st.tabs(["Revenue Opportunity","Feature Priority Matrix","Segment Strategy"])

    with tab1:
        st.markdown("#### Revenue potential by cluster")
        wtp_value = {'Free only':0,'Maybe':0,'<99':99,'<299':299,'<499':499,'>499':799}
        df['wtp_value'] = df['Q23_wtp'].map(wtp_value)

        rev_data = df.groupby('cluster').agg(
            Segment_Size=('cluster','count'),
            Likely_Adopters=('Q25_adoption_3class', lambda x: (x=='Likely').sum()),
            Avg_WTP_INR=('wtp_value','mean'),
            Conversion_Rate=('Q25_adoption_3class', lambda x: (x=='Likely').mean()),
        ).round(2)
        rev_data['Est_Monthly_Rev_INR'] = (rev_data['Likely_Adopters'] *
                                            rev_data['Avg_WTP_INR'] *
                                            rev_data['Conversion_Rate']).round(0)
        rev_data['Segment_Size_Extrapolated'] = (rev_data['Segment_Size'] * 40000).astype(int)

        st.dataframe(rev_data.style
                     .background_gradient(cmap='Greens', subset=['Est_Monthly_Rev_INR'])
                     .format({'Avg_WTP_INR':'₹{:.0f}','Conversion_Rate':'{:.1%}',
                              'Est_Monthly_Rev_INR':'₹{:,.0f}'}),
                     use_container_width=True)

        fig = px.bar(rev_data.reset_index(), x='cluster', y='Est_Monthly_Rev_INR',
                     color='Est_Monthly_Rev_INR', color_continuous_scale='Greens',
                     title='Estimated monthly revenue potential per cluster (survey sample)',
                     labels={'cluster':'Cluster','Est_Monthly_Rev_INR':'Est. Monthly Revenue (₹)'})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        total_likely = (df['Q25_adoption_3class']=='Likely').sum()
        avg_wtp = df[df['Q25_adoption_3class']=='Likely']['wtp_value'].mean()
        st.markdown(f'<div class="insight-box">💰 Based on survey sample: <b>{total_likely} likely adopters</b> with average WTP of <b>₹{avg_wtp:.0f}/month</b>. Extrapolating to India\'s ~80M fintech-ready population, even 0.5% conversion at this WTP represents ₹{0.005*80000000*avg_wtp/10000000:.0f} Cr+ monthly ARR potential.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Feature priority matrix — demand vs WTP correlation")
        feat_priority = []
        for col, label in FEATURE_LABELS.items():
            demand = df[col].mean()
            avg_wtp_feat = df[df[col]==1]['Q23_wtp_numeric'].mean()
            likely_rate = df[df[col]==1]['Q25_adoption_3class'].value_counts(normalize=True).get('Likely',0)
            feat_priority.append({'Feature': label,'Demand': demand,
                                   'Avg WTP when selected': avg_wtp_feat,
                                   'Adoption rate': likely_rate})
        fp_df = pd.DataFrame(feat_priority)

        fig = px.scatter(fp_df, x='Demand', y='Avg WTP when selected',
                         size='Adoption rate', color='Adoption rate',
                         text='Feature', color_continuous_scale='Purples',
                         title='Feature priority matrix — Demand vs WTP signal (bubble = adoption rate)',
                         size_max=40)
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.add_vline(x=fp_df['Demand'].mean(), line_dash='dash', line_color='grey')
        fig.add_hline(y=fp_df['Avg WTP when selected'].mean(), line_dash='dash', line_color='grey')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Interpretation:** Top-right quadrant = high demand AND high WTP signal → MVP must-haves. Top-left = premium features. Bottom-right = free tier hooks.")

    with tab3:
        st.markdown("#### Segment-level marketing strategy")
        seg_strategy = {
            0: {
                "name": "Digital Natives",
                "channel": "Instagram Reels, YouTube Shorts, Discord",
                "message": "Invest smarter in 2 minutes. AI picks. You grow.",
                "tier": "Free → Premium (quick conversion)",
                "priority": "App-first, gamification, social sharing",
            },
            1: {
                "name": "Cautious Savers",
                "channel": "WhatsApp, Email, TV ads",
                "message": "Your money, protected and growing. SEBI registered.",
                "tier": "Free with trust-building content",
                "priority": "Trust signals, FD-equivalent returns, slow onboarding",
            },
            2: {
                "name": "Wealth Builders",
                "channel": "LinkedIn, Business press, Referral",
                "message": "One platform for your entire financial life.",
                "tier": "Premium / Wealth tier from day 1",
                "priority": "Advanced analytics, tax harvesting, advisor access",
            },
            3: {
                "name": "Aspiring Beginners",
                "channel": "YouTube, Tier-2 city partnerships, Employer tie-ups",
                "message": "Start with ₹500/month. Your first SIP in 3 minutes.",
                "tier": "Free tier — long nurture",
                "priority": "Education content, goal wizard, low minimums",
            },
            4: {
                "name": "Delegation Seekers",
                "channel": "Advisor network, Bank partnerships, Word of mouth",
                "message": "Tell us your goals. We handle everything.",
                "tier": "Premium / Advisory",
                "priority": "Human advisor integration, full robo delegation",
            },
        }
        for i, s in seg_strategy.items():
            cluster_size = (df['cluster']==i).sum()
            likely_pct = (df[df['cluster']==i]['Q25_adoption_3class']=='Likely').mean()*100
            color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
            st.markdown(f"""
            <div style="border-left:4px solid {color};border-radius:0 12px 12px 0;
                        padding:1rem 1.25rem;margin-bottom:1rem;background:var(--background-color,#f8f9fa)">
              <div style="color:{color};font-weight:600;font-size:1rem">Cluster {i} — {s['name']}
                <span style="font-weight:400;font-size:0.85rem;color:#666;margin-left:8px">
                  n={cluster_size} · {likely_pct:.0f}% likely adopters</span></div>
              <div style="margin-top:8px;font-size:0.88rem;display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div>📢 <b>Channel:</b> {s['channel']}</div>
                <div>💬 <b>Message:</b> <i>{s['message']}</i></div>
                <div>💳 <b>Recommended tier:</b> {s['tier']}</div>
                <div>🎯 <b>Focus:</b> {s['priority']}</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — NEW CUSTOMER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀 New Customer Predictor":
    st.title("🚀 New Customer Predictor")
    st.markdown("*Upload new survey responses — get instant adoption predictions and marketing recommendations*")
    st.divider()

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    @st.cache_resource
    def train_all_models():
        X, feat_cols = get_ml_features(df)
        y_clf = df['Q25_adoption_3class']
        y_reg = df['Q23_wtp_numeric'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        le = LabelEncoder()
        y_enc = le.fit_transform(y_clf)

        clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        clf.fit(X_scaled, y_enc)

        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_scaled, y_reg)

        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        km.fit(X_scaled)

        return clf, reg, km, scaler, le, feat_cols

    clf, reg, km, scaler, le, feat_cols = train_all_models()

    st.markdown("### Upload new respondent data")
    st.markdown("Upload a CSV with the same column format as the training data. The model will predict adoption class, WTP tier, and cluster for each respondent.")

    new_file = st.file_uploader("Upload new survey CSV", type=['csv'], key='predictor')

    st.divider()
    st.markdown("#### Or try a single respondent manually")

    with st.expander("Fill in respondent details"):
        c1,c2,c3 = st.columns(3)
        m_age     = c1.selectbox("Age group", AGE_ORDER)
        m_income  = c2.selectbox("Income", INCOME_ORDER)
        m_edu     = c3.selectbox("Education", ['Up to 12th','Graduate','Postgraduate','Professional','Doctorate'])
        m_region  = c1.selectbox("Region", ['Metro','Tier-2','Tier-3','Rural'])
        m_comfort = c2.slider("App comfort (1-5)", 1, 5, 3)
        m_nps     = c3.slider("NPS score (1-10)", 1, 10, 7)
        m_tax     = c1.slider("Tax importance (1-5)", 1, 5, 3)
        m_wtp     = c2.selectbox("WTP tier", WTP_ORDER)
        m_sharing = c3.selectbox("Data sharing", ['No-privacy','Basic only','Yes-if-regulated','Yes-fully'])
        m_risk    = c1.selectbox("Risk appetite", RISK_ORDER)
        m_invest  = c2.selectbox("Monthly investment", ['Nothing','<1K','1K-5K','5K-15K','15K-50K','>50K'])
        m_horizon = c3.selectbox("Time horizon", ['<1yr','1-3yr','3-7yr','7-15yr','15+yr'])

        if st.button("Predict this respondent"):
            # Build a minimal row matching the feature set
            sample = pd.DataFrame([{c: 0 for c in df.columns if c != 'respondent_id'}])
            sample['Q1_age'] = m_age
            sample['Q5_income'] = m_income
            sample['Q3_education'] = m_edu
            sample['Q2_region'] = m_region
            sample['Q20_app_comfort'] = m_comfort
            sample['Q24_nps'] = m_nps
            sample['Q19_tax_importance'] = m_tax
            sample['Q23_wtp'] = m_wtp
            sample['Q23_wtp_numeric'] = WTP_ORDER.index(m_wtp)
            sample['Q21_data_sharing'] = m_sharing
            sample['Q9_risk_appetite'] = m_risk
            sample['Q7_monthly_investment'] = m_invest
            sample['Q9_time_horizon'] = m_horizon
            sample['Q4_occupation'] = 'Salaried-Private'
            sample['Q11_dropout'] = 'Never tried'
            sample['Q12_loss_reaction'] = 'Hold'
            sample['Q13_loss_aversion'] = 'No preference'
            sample['Q14_autonomy'] = 'Guided'
            sample['Q15_influence'] = 'Myself'
            sample['Q25_adoption_3class'] = 'Neutral'
            sample['Q25_adoption_5class'] = 'Neutral'
            sample['Q25_adoption_score'] = 0
            sample['Q20_pct_invest'] = '10-20%'

            X_s, _ = get_ml_features(sample)
            X_s = X_s.reindex(columns=feat_cols, fill_value=0)
            X_s_scaled = scaler.transform(X_s)

            pred_class = le.inverse_transform(clf.predict(X_s_scaled))[0]
            pred_proba = clf.predict_proba(X_s_scaled)[0]
            pred_wtp   = reg.predict(X_s_scaled)[0]
            pred_cluster = km.predict(X_s_scaled)[0]

            color = PERSONA_COLORS.get(pred_class, '#888')
            st.markdown(f"""
            <div style="border:2px solid {color};border-radius:12px;padding:1.5rem;margin-top:1rem">
              <div style="font-size:1.2rem;font-weight:600;color:{color}">
                Adoption prediction: {pred_class}</div>
              <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:0.9rem">
                <div>📊 <b>Cluster:</b> {pred_cluster}</div>
                <div>💰 <b>Predicted WTP tier:</b> {min(round(pred_wtp), 5)}/5</div>
                <div>🎯 <b>Confidence:</b> {max(pred_proba)*100:.1f}%</div>
              </div>
              <div style="margin-top:12px;font-size:0.85rem">
                <b>Class probabilities:</b>
                {"  |  ".join([f"{le.classes_[i]}: {pred_proba[i]*100:.1f}%" for i in range(len(le.classes_))])}
              </div>
            </div>""", unsafe_allow_html=True)

    if new_file is not None:
        st.divider()
        st.markdown("### Batch prediction results")
        new_df = pd.read_csv(new_file)
        st.info(f"Loaded {len(new_df)} new respondents")

        try:
            X_new, _ = get_ml_features(new_df)
            X_new = X_new.reindex(columns=feat_cols, fill_value=0)
            X_new_s = scaler.transform(X_new)

            new_df['Predicted_Adoption'] = le.inverse_transform(clf.predict(X_new_s))
            probas = clf.predict_proba(X_new_s)
            for i, cls in enumerate(le.classes_):
                new_df[f'Prob_{cls}'] = (probas[:,i]*100).round(1)
            new_df['Predicted_WTP_Tier'] = reg.predict(X_new_s).round(1)
            new_df['Cluster'] = km.predict(X_new_s)

            st.markdown("#### Adoption distribution in new data")
            vc = new_df['Predicted_Adoption'].value_counts()
            fig = px.pie(values=vc.values, names=vc.index,
                         color=vc.index, color_discrete_map=PERSONA_COLORS, hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Scored respondents")
            display_cols = [c for c in ['respondent_id','Q1_age','Q5_income','Q2_region',
                                         'Predicted_Adoption','Predicted_WTP_Tier','Cluster'] +
                            [f'Prob_{c}' for c in le.classes_] if c in new_df.columns]
            st.dataframe(new_df[display_cols].style
                         .applymap(lambda v: f"background-color: {PERSONA_COLORS.get(v,'')}"
                                   if isinstance(v,str) and v in PERSONA_COLORS else '',
                                   subset=['Predicted_Adoption']),
                         use_container_width=True)

            csv_out = new_df.to_csv(index=False)
            st.download_button("Download scored CSV", csv_out,
                               "scored_new_customers.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}. Please ensure the CSV matches the training data format.")
