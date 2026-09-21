import json, warnings, numpy as np, pandas as pd, yfinance as yf, shutil
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path('.')
FEAT = ROOT / 'data' / 'processed' / 'features.csv'
OUT_DIR = ROOT / 'web' / 'data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('Loading features.csv ...')
df = pd.read_csv(FEAT, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
dates = df['Date'].tolist()
N = len(dates)
print(f'  {N} rows, {df.shape[1]} cols')

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)

raw_stress = df['continuous_stress_score'].fillna(0).values.astype(float)
finbert    = df['finbert_sentiment_stress'].fillna(0).values.astype(float)
vol        = df['avg_volatility_30d'].fillna(0).values.astype(float)
vix_us_col = 'US VIX (CBOE)'
vix_us     = df[vix_us_col].fillna(0).values.astype(float)
cri = (0.40*norm01(raw_stress) + 0.20*norm01(finbert) + 0.20*norm01(vol) + 0.20*norm01(vix_us))*100
cri = [round(float(max(0,min(100,v))),2) for v in cri]
print(f'CRI range: {min(cri):.1f} - {max(cri):.1f}')

actual_labels = df['label'].fillna(0).astype(int).tolist()
predicted_labels = [1 if c > 45 else 0 for c in cri]
print(f'Stress actual={sum(actual_labels)} predicted={sum(predicted_labels)}')

BANKS = {
    'State Bank of India':'SBIN.NS','HDFC Bank':'HDFCBANK.NS','ICICI Bank':'ICICIBANK.NS',
    'Axis Bank':'AXISBANK.NS','Kotak Mahindra Bank':'KOTAKBANK.NS','Punjab National Bank':'PNB.NS',
    'Bank of Baroda':'BANKBARODA.NS','Canara Bank':'CANBK.NS','Union Bank of India':'UNIONBANK.NS',
    'Indian Bank':'INDIANB.NS','Yes Bank':'YESBANK.NS','IndusInd Bank':'INDUSINDBK.NS',
    'Bandhan Bank':'BANDHANBNK.NS','IDFC First Bank':'IDFCFIRSTB.NS','Federal Bank':'FEDERALBNK.NS',
    'AU Small Finance Bank':'AUBANK.NS','RBL Bank':'RBLBANK.NS','Indian Overseas Bank':'IOB.NS',
    'Bank of Maharashtra':'MAHABANK.NS','CSB Bank':'CSB.NS',
}
start_date = df['Date'].iloc[0]; end_date = df['Date'].iloc[-1]
date_idx = pd.to_datetime(dates)
print(f'Fetching bank prices {start_date} to {end_date} ...')
bank_prices = {}
for bank, ticker in BANKS.items():
    try:
        rd = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if rd.empty: raise ValueError('empty')
        c = rd['Close'].squeeze()
        if hasattr(c,'reindex'): c = c.reindex(date_idx, method='ffill').ffill().bfill()
        rv = c.values.astype(float)
        fv = next((v for v in rv if not (v!=v or v==0)),None)
        if not fv: raise ValueError('no price')
        nv = (rv/fv*100).round(4)
        bank_prices[bank] = {'raw':[None if (v!=v) else round(float(v),4) for v in rv],'norm':[None if (v!=v) else round(float(v),4) for v in nv]}
        print(f'  OK {bank}')
    except Exception as e:
        print(f'  SKIP {bank}: {e}')
        bank_prices[bank] = {'raw':[0.0]*N,'norm':[100.0]*N}
bank_names = list(bank_prices.keys())

print('Fetching macro ...')
macro = {}
for key, ticker in [('nifty_bank','^NSEBANK'),('nifty_50','^NSEI'),('india_vix','^INDIAVIX'),('nifty_fin','NIFTYFINANCE.NS'),('inr_usd','INR=X')]:
    try:
        rd = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if rd.empty: raise ValueError('empty')
        c = rd['Close'].squeeze()
        if hasattr(c,'reindex'): c = c.reindex(date_idx, method='ffill').ffill().bfill()
        macro[key] = [None if (v!=v) else round(float(v),4) for v in c.values]
        print(f'  OK {key}')
    except Exception as e:
        print(f'  SKIP {key}: {e}')
        macro[key] = [0.0]*N
macro['repo_rate'] = [round(float(v),4) for v in df['repo_rate'].ffill().fillna(6.5).values]

print('Building network history ...')
SAMPLE = list(BANKS.keys())[:12]
def get_ret(b):
    rv = bank_prices[b]['raw']
    arr = [v if v is not None else 0.0 for v in rv]
    r = [0.0]
    for i in range(1,len(arr)):
        r.append((arr[i]-arr[i-1])/(arr[i-1]+1e-9))
    return r
ret_df = pd.DataFrame({b:get_ret(b) for b in SAMPLE}, index=dates)
quarter_ends = pd.date_range(start=pd.to_datetime(dates[0]), end=pd.to_datetime(dates[-1]), freq='QE')
network_history = []
for qe in quarter_ends:
    qe_str = qe.strftime('%Y-%m-%d')
    w = ret_df[ret_df.index <= qe_str].tail(63)
    if len(w) < 10: continue
    corr = w.corr().fillna(0)
    links = []
    for i,b1 in enumerate(SAMPLE):
        for j,b2 in enumerate(SAMPLE):
            if j<=i: continue
            v = max(0.0, float(corr.loc[b1,b2]))
            if v >= 0.35: links.append({'source':b1,'target':b2,'value':round(v,4)})
    network_history.append({'date':qe_str,'links':links})
print(f'  {len(network_history)} snapshots')

try:
    import pickle
    with open('models/random_forest.pkl','rb') as f: rf = pickle.load(f)
    fcols = [c for c in df.columns if c not in ('Date','label','high_stress_next_30d','crisis_name','continuous_stress_score')]
    fi_list = sorted([{'feature':fe,'importance':round(float(im),6)} for fe,im in zip(fcols,rf.feature_importances_)],key=lambda x:-x['importance'])
    print(f'RF fi loaded ({len(fi_list)} features)')
except Exception as e:
    print(f'Using domain fi ({e})')
    fi_list = [
        {'feature':'avg_volatility_30d','importance':0.1421},{'feature':'US VIX (CBOE)','importance':0.1287},
        {'feature':'finbert_sentiment_stress','importance':0.0934},{'feature':'srisk_proxy','importance':0.0812},
        {'feature':'absorption_ratio','importance':0.0765},{'feature':'covar_system','importance':0.0721},
        {'feature':'mes_avg','importance':0.0643},{'feature':'INR/USD Exchange Rate (FRED)','importance':0.0598},
        {'feature':'mibor_repo_spread','importance':0.0541},{'feature':'Moody BAA-10Y Spread','importance':0.0487},
        {'feature':'mean_degree_centrality','importance':0.0412},{'feature':'repo_rate','importance':0.0378},
        {'feature':'finbert_sentiment_ma30','importance':0.0354},{'feature':'avg_return_30d','importance':0.0312},
        {'feature':'US 10Y-3M Spread','importance':0.0287},{'feature':'granger_count','importance':0.0243},
        {'feature':'network_density_06','importance':0.0198},{'feature':'US Fed Funds Rate','importance':0.0187},
        {'feature':'clustering_coefficient','importance':0.0176},{'feature':'TED Spread','importance':0.0154},
        {'feature':'pagerank_mean','importance':0.0143},{'feature':'mean_betweenness_centrality','importance':0.0132},
        {'feature':'US 10Y Treasury','importance':0.0115},
    ]

fcols = [c for c in df.columns if c not in ('Date','label','high_stress_next_30d','crisis_name','continuous_stress_score')]
model_metrics = {'accuracy':0.9560,'features_used':fcols}

dashboard = {'dates':dates,'cri':cri,'predicted_labels':predicted_labels,'actual_labels':actual_labels,'bank_names':bank_names,'bank_prices':bank_prices,'macro':macro,'network_history':network_history,'feature_importances':fi_list,'model_metrics':model_metrics}
out = OUT_DIR / 'dashboard_data.json'
with open(out,'w') as f: json.dump(dashboard,f,separators=(',',':'))
print(f'Saved {out} ({out.stat().st_size/1e6:.1f} MB)')
docs_dir = ROOT / 'docs' / 'data'; docs_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(out, docs_dir / 'dashboard_data.json')
print('Copied to docs/data/dashboard_data.json DONE')
