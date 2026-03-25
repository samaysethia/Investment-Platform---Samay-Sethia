import pandas as pd
import numpy as np

# ── Ordinal encoders ──────────────────────────────────────────────────────────
AGE_ORDER       = ['18-24','25-34','35-44','45-54','55+']
INCOME_ORDER    = ['<3L','3-6L','6-12L','12-25L','25-50L','>50L']
EDU_ORDER       = ['Up to 12th','Graduate','Postgraduate','Professional','Doctorate']
MONINV_ORDER    = ['Nothing','<1K','1K-5K','5K-15K','15K-50K','>50K']
HORIZON_ORDER   = ['<1yr','1-3yr','3-7yr','7-15yr','15+yr']
RISK_ORDER      = ['Very Conservative','Conservative','Moderate','Aggressive','Very Aggressive']
WTP_ORDER       = ['Free only','Maybe','<99','<299','<499','>499']
PCTINV_ORDER    = ['<5%','5-10%','10-20%','20-30%','>30%']
ADOPTION5_ORDER = ['Definitely will not use','Unlikely to use','Neutral','Likely to use','Definitely will use']
ADOPTION3_ORDER = ['Unlikely','Neutral','Likely']

ORDINAL_MAPS = {
    'Q1_age':            {v:i for i,v in enumerate(AGE_ORDER)},
    'Q5_income':         {v:i for i,v in enumerate(INCOME_ORDER)},
    'Q3_education':      {v:i for i,v in enumerate(EDU_ORDER)},
    'Q7_monthly_investment': {v:i for i,v in enumerate(MONINV_ORDER)},
    'Q9_time_horizon':   {v:i for i,v in enumerate(HORIZON_ORDER)},
    'Q9_risk_appetite':  {v:i for i,v in enumerate(RISK_ORDER)},
    'Q23_wtp':           {v:i for i,v in enumerate(WTP_ORDER)},
    'Q20_pct_invest':    {v:i for i,v in enumerate(PCTINV_ORDER)},
    'Q25_adoption_5class': {v:i for i,v in enumerate(ADOPTION5_ORDER)},
    'Q25_adoption_3class': {v:i for i,v in enumerate(ADOPTION3_ORDER)},
}

NOMINAL_COLS = ['Q2_region','Q4_occupation','Q11_dropout','Q12_loss_reaction',
                'Q13_loss_aversion','Q14_autonomy','Q15_influence','Q21_data_sharing']

BINARY_COLS = [c for c in [
    'goal_retirement','goal_child_edu','goal_home','goal_emergency',
    'goal_tax_saving','goal_wealth','goal_travel','goal_business','goal_debt',
    'inv_fd','inv_mf','inv_equity','inv_ppf_epf','inv_nps','inv_gold',
    'inv_realestate','inv_insurance','inv_crypto','inv_none',
    'plat_zerodha','plat_groww','plat_etmoney','plat_paytmmoney','plat_bankapp',
    'plat_indmoney','plat_kuvera','plat_excel','plat_advisor','plat_none',
    'pain_no_start','pain_confusing','pain_no_time','pain_fear_loss','pain_high_min',
    'pain_no_advice','pain_tax','pain_bad_app','pain_no_trust',
    'switch_better_reco','switch_lower_fees','switch_simpler_ux','switch_more_products',
    'switch_human_advisor','switch_security','switch_loyal','switch_no_platform',
    'feat_goal_planning','feat_ai_reco','feat_tax_tool','feat_one_platform',
    'feat_sip_auto','feat_networth','feat_rebalance','feat_baskets',
    'feat_education','feat_human_advisor','feat_family','feat_insurance',
    'trust_sebi','trust_security','trust_brand','trust_reviews',
    'trust_fees','trust_trial','trust_media',
]]

NUMERIC_COLS = ['Q19_tax_importance','Q20_app_comfort','Q23_wtp_numeric',
                'Q24_nps','Q25_adoption_score']

FEATURE_LABELS = {
    'feat_goal_planning':'Goal planning','feat_ai_reco':'AI recommendations',
    'feat_tax_tool':'Tax tools','feat_one_platform':'One platform',
    'feat_sip_auto':'SIP automation','feat_networth':'Net worth dashboard',
    'feat_rebalance':'Rebalancing alerts','feat_baskets':'Thematic baskets',
    'feat_education':'Financial education','feat_human_advisor':'Human advisor',
    'feat_family':'Family account','feat_insurance':'Insurance integration',
}
GOAL_LABELS = {
    'goal_retirement':'Retirement','goal_child_edu':'Child education',
    'goal_home':'Buy home','goal_emergency':'Emergency fund',
    'goal_tax_saving':'Tax saving','goal_wealth':'Wealth creation',
    'goal_travel':'Travel/lifestyle','goal_business':'Business funding','goal_debt':'Pay off debt',
}
PAIN_LABELS = {
    'pain_no_start':'No idea where to start','pain_confusing':'Too confusing',
    'pain_no_time':'No time to monitor','pain_fear_loss':'Fear of loss',
    'pain_high_min':'High minimums','pain_no_advice':'No personalised advice',
    'pain_tax':'Difficult tax filing','pain_bad_app':'Poor app experience',
    'pain_no_trust':'Lack of trust',
}
INV_LABELS = {
    'inv_fd':'Fixed Deposits','inv_mf':'Mutual Funds','inv_equity':'Direct Equity',
    'inv_ppf_epf':'PPF/EPF','inv_nps':'NPS','inv_gold':'Gold',
    'inv_realestate':'Real Estate','inv_insurance':'Insurance (ULIP/LIC)',
    'inv_crypto':'Crypto','inv_none':'None currently',
}

PERSONA_COLORS = {
    'Likely':   '#1D9E75',
    'Neutral':  '#EF9F27',
    'Unlikely': '#D85A30',
}

CLUSTER_PALETTE = ['#534AB7','#1D9E75','#D85A30','#BA7517','#185FA5','#993556']


def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        import os
        path = os.path.join(os.path.dirname(__file__), 'investment_survey_data.csv')
        df = pd.read_csv(path)
    return df


def encode_for_ml(df):
    dfc = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in dfc.columns:
            dfc[col+'_enc'] = dfc[col].map(mapping)
    for col in NOMINAL_COLS:
        if col in dfc.columns:
            dummies = pd.get_dummies(dfc[col], prefix=col, drop_first=True)
            dfc = pd.concat([dfc, dummies], axis=1)
    return dfc


def get_ml_features(df):
    dfe = encode_for_ml(df)
    feature_cols = (
        [c+'_enc' for c in ORDINAL_MAPS if c+'_enc' in dfe.columns
         and c not in ['Q25_adoption_5class','Q25_adoption_3class','Q23_wtp']] +
        [c for c in dfe.columns if any(c.startswith(n+'_') for n in NOMINAL_COLS)] +
        BINARY_COLS +
        ['Q19_tax_importance','Q20_app_comfort','Q24_nps']
    )
    feature_cols = [c for c in feature_cols if c in dfe.columns]
    feature_cols = list(dict.fromkeys(feature_cols))
    return dfe[feature_cols].fillna(0), feature_cols
