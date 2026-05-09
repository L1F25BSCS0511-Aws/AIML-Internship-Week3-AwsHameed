from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
OUT=Path(__file__).resolve().parent
print('loading')
df=pd.read_csv(OUT/'train.csv')
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#F8FAFC','axes.grid':True,'grid.alpha':.25,'grid.linestyle':'--','font.size':10,'savefig.dpi':150,'savefig.bbox':'tight'})
sns.set_theme(style='whitegrid')
# feature engineering
fe=df.copy(); fe['TotalSF']=fe['TotalBsmtSF'].fillna(0)+fe['1stFlrSF'].fillna(0)+fe['2ndFlrSF'].fillna(0); fe['TotalBaths']=fe['FullBath'].fillna(0)+0.5*fe['HalfBath'].fillna(0)+fe['BsmtFullBath'].fillna(0)+0.5*fe['BsmtHalfBath'].fillna(0); fe['HouseAge']=fe['YrSold']-fe['YearBuilt']; fe['RemodelAge']=fe['YrSold']-fe['YearRemodAdd']; fe['HasRemodeled']=(fe['YearBuilt']!=fe['YearRemodAdd']).astype(int); fe['QualCond']=fe['OverallQual']*fe['OverallCond']; fe['PricePerSF']=np.where(fe['TotalSF']>0,fe['SalePrice']/fe['TotalSF'],np.nan); fe['IsNewHouse']=(fe['YearBuilt']>=fe['YrSold']-5).astype(int)
new_features=['TotalSF','TotalBaths','HouseAge','RemodelAge','HasRemodeled','QualCond','PricePerSF','IsNewHouse']
# encoding
cat_cols=fe.select_dtypes(include='object').columns.tolist(); quality_cols=['ExterQual','KitchenQual','BsmtQual','GarageQual','FireplaceQu']; records=[]
for col in cat_cols:
    missing=fe[col].isna().mean(); dom=fe[col].value_counts(normalize=True,dropna=False).iloc[0]; nunique=fe[col].nunique(dropna=True)
    if col in quality_cols: strategy='label_encode'; reason='Ordinal quality scale'
    elif missing>.5 or dom>.95: strategy='drop'; reason='>50% missing or >95% dominant'
    elif nunique>10: strategy='frequency_encode'; reason='High-cardinality nominal'
    else: strategy='onehot_encode'; reason='Nominal <=10 categories'
    records.append({'column':col,'unique_categories':nunique,'missing_%':round(missing*100,2),'dominant_%':round(dom*100,2),'strategy':strategy,'reason':reason})
strategy_table=pd.DataFrame(records); strategy_table.to_csv(OUT/'week3_encoding_decision_table.csv', index=False)
encoded=fe.copy(); qmap={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
label_cols=strategy_table.query("strategy=='label_encode'")['column'].tolist(); drop_cols=strategy_table.query("strategy=='drop'")['column'].tolist(); freq_cols=strategy_table.query("strategy=='frequency_encode'")['column'].tolist(); onehot_cols=strategy_table.query("strategy=='onehot_encode'")['column'].tolist()
for col in label_cols: encoded[col]=encoded[col].fillna('NA').map(qmap).fillna(0).astype(int)
encoded=encoded.drop(columns=drop_cols,errors='ignore')
for col in freq_cols:
    vals=encoded[col].fillna('NA'); freq=vals.value_counts(normalize=True); encoded[col+'_freq']=vals.map(freq); encoded=encoded.drop(columns=[col])
for col in onehot_cols:
    if col in encoded.columns: encoded[col]=encoded[col].fillna('NA')
encoded=pd.get_dummies(encoded, columns=[c for c in onehot_cols if c in encoded.columns], drop_first=True, dtype=int)
num_cols=encoded.select_dtypes(include='number').columns; encoded[num_cols]=encoded[num_cols].fillna(encoded[num_cols].median())
print('encoded', encoded.shape)
# scaling
X=encoded.drop('SalePrice',axis=1); y=encoded['SalePrice']; X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
scalers={'StandardScaler':StandardScaler(),'MinMaxScaler':MinMaxScaler(),'RobustScaler':RobustScaler()}; scaled_data={}; rows=[]
for name,scaler in scalers.items():
    tr=scaler.fit_transform(X_train); te=scaler.transform(X_test); trdf=pd.DataFrame(tr,columns=X_train.columns,index=X_train.index); tedf=pd.DataFrame(te,columns=X_test.columns,index=X_test.index); scaled_data[name]=(trdf,tedf,scaler)
    for feat in ['GrLivArea','TotalSF','LotArea']: rows.append({'scaler':name,'feature':feat,'raw_mean':X_train[feat].mean(),'raw_std':X_train[feat].std(),'scaled_mean':trdf[feat].mean(),'scaled_std':trdf[feat].std()})
pd.DataFrame(rows).to_csv(OUT/'week3_scaling_comparison_table.csv', index=False)
# skewness and transformations
skew=encoded.select_dtypes(include='number').skew().dropna()
def severity(v):
    av=abs(v); return 'Normal' if av<.5 else 'Moderate' if av<1 else 'High' if av<2 else 'Very High'
skew_table=pd.DataFrame({'column':skew.index,'skewness':skew.values,'severity':[severity(v) for v in skew.values]}).sort_values('skewness', key=lambda s:s.abs(), ascending=False); skew_table.to_csv(OUT/'week3_skewness_table.csv',index=False)
trans=encoded.copy(); orig_skew=trans.select_dtypes(include='number').skew().dropna(); treat_cols=[c for c,v in orig_skew.items() if c!='SalePrice' and abs(v)>.75 and trans[c].min()>=0 and trans[c].nunique()>1]
print('treat cols', len(treat_cols))
for c in treat_cols: trans[c]=np.log1p(trans[c])
sp=encoded['SalePrice']; sale_skew=sp.skew(); sp_log=np.log1p(sp); log_sale_skew=pd.Series(sp_log).skew(); sp_sqrt=np.sqrt(sp); sp_bc,lamb=stats.boxcox(sp)
transform_summary=pd.DataFrame({'transformation':['Original','log1p','sqrt','Box-Cox'],'skewness':[sp.skew(),pd.Series(sp_log).skew(),pd.Series(sp_sqrt).skew(),pd.Series(sp_bc).skew()]}); transform_summary.to_csv(OUT/'week3_saleprice_transformation_comparison.csv',index=False)
best=transform_summary.iloc[transform_summary['skewness'].abs().argmin()]['transformation']; trans['SalePrice_transformed']=sp_bc if best=='Box-Cox' else sp_log if best=='log1p' else sp_sqrt if best=='sqrt' else sp
print('best', best)
# skew before-after figure
plot_features=[c for c in treat_cols if trans[c].nunique()>20 and c!='Id'][:3]
if len(plot_features)<3:
    for c in treat_cols:
        if c not in plot_features and c!='Id': plot_features.append(c)
        if len(plot_features)==3: break
fig, axes=plt.subplots(4,2,figsize=(14,16))
for i,col in enumerate(['SalePrice']+plot_features[:3]):
    if col=='SalePrice': original=encoded[col]; transformed=trans['SalePrice_transformed']; after=f'{best} transformed SalePrice'
    else: original=encoded[col]; transformed=trans[col]; after=f'log1p({col})'
    axes[i,0].hist(original,bins=35,color='#5B21B6',edgecolor='white',alpha=.8); axes[i,0].set_title(f'Original {col} — skew={pd.Series(original).skew():.2f}')
    axes[i,1].hist(transformed,bins=35,color='#0D9488',edgecolor='white',alpha=.8); axes[i,1].set_title(f'{after} — skew={pd.Series(transformed).skew():.2f}')
fig.suptitle('Skewness Treatment: Before vs After',fontsize=16,fontweight='bold'); plt.tight_layout(); plt.savefig(OUT/'w3_skewness_before_after.png'); plt.close(fig)
print('saved skew before after')
# feature selection
feature_candidates=trans.drop(columns=['SalePrice','SalePrice_transformed','Id','PricePerSF'],errors='ignore'); target=trans['SalePrice']; feature_corr=feature_candidates.corrwith(target).dropna(); top30=feature_corr.abs().sort_values(ascending=False).head(30).index.tolist(); X_top=feature_candidates[top30].copy(); vt=VarianceThreshold(threshold=.01); X_var=pd.DataFrame(vt.fit_transform(X_top), columns=X_top.columns[vt.get_support()], index=X_top.index)
cm=X_var.corr().abs(); upper=cm.where(np.triu(np.ones(cm.shape),k=1).astype(bool)); to_drop=[]
for col in upper.columns:
    for row in upper.index[upper[col]>.95].tolist():
        drop=col if abs(feature_corr.get(col,0))<abs(feature_corr.get(row,0)) else row
        if drop not in to_drop: to_drop.append(drop)
X_final=X_var.drop(columns=to_drop,errors='ignore'); final_features_info=pd.DataFrame({'feature':X_final.columns,'correlation_with_SalePrice':feature_corr[X_final.columns].values,'dtype':[str(feature_candidates[c].dtype) for c in X_final.columns]}).sort_values('correlation_with_SalePrice', key=lambda s:s.abs(), ascending=False); final_features_info.to_csv(OUT/'week3_final_selected_features.csv', index=False)
print('final features', len(final_features_info))
# dashboard
fig, axes=plt.subplots(3,2,figsize=(16,18)); axes=axes.ravel()
raw_z=pd.Series(stats.zscore(encoded['SalePrice'])); log_z=pd.Series(stats.zscore(np.log1p(encoded['SalePrice'])))
sns.histplot(raw_z,bins=35,kde=True,stat='density',color='#5B21B6',alpha=.35,ax=axes[0],label='Raw'); sns.histplot(log_z,bins=35,kde=True,stat='density',color='#0D9488',alpha=.35,ax=axes[0],label='Log1p'); axes[0].set_title('1) SalePrice Distribution Before vs After log1p'); axes[0].set_xlabel('Standardized value'); axes[0].text(.03,.95,f"Raw skew={encoded['SalePrice'].skew():.2f}\nLog skew={np.log1p(encoded['SalePrice']).skew():.2f}", transform=axes[0].transAxes, va='top', bbox=dict(boxstyle='round',facecolor='white',alpha=.85)); axes[0].legend()
sc=axes[1].scatter(encoded['TotalSF'],encoded['SalePrice'],c=encoded['OverallQual'],cmap='RdYlGn',alpha=.6,s=35); coef=np.polyfit(encoded['TotalSF'],encoded['SalePrice'],1); line_x=np.linspace(encoded['TotalSF'].min(),encoded['TotalSF'].max(),200); axes[1].plot(line_x,np.poly1d(coef)(line_x),color='#111827',lw=2); axes[1].set(title='2) TotalSF vs SalePrice',xlabel='TotalSF',ylabel='SalePrice'); axes[1].text(.03,.95,f"r={encoded['TotalSF'].corr(encoded['SalePrice']):.3f}",transform=axes[1].transAxes,va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=.85)); fig.colorbar(sc,ax=axes[1],label='OverallQual')
corr_dash=trans.drop(columns=['SalePrice_transformed'],errors='ignore').corr()['SalePrice'].drop('SalePrice').sort_values(key=lambda s:s.abs(),ascending=False).head(15).sort_values(); colors=['#16A34A' if v>0 else '#DC2626' for v in corr_dash.values]; axes[2].barh(corr_dash.index,corr_dash.values,color=colors); axes[2].set(title='3) Top 15 Correlation-Based Feature Importances',xlabel='Correlation with SalePrice'); axes[2].axvline(0,color='black',lw=1)
sns.boxplot(data=encoded,x='OverallQual',y='SalePrice',palette='viridis',ax=axes[3]); axes[3].set(title='4) SalePrice by Overall Quality',xlabel='OverallQual',ylabel='SalePrice'); med=encoded.groupby('OverallQual')['SalePrice'].median()
for i,(q,m) in enumerate(med.items()): axes[3].text(i,m,f'{m/1000:.0f}K',ha='center',va='bottom',fontsize=8,color='black')
heat_cols=trans.drop(columns=['SalePrice_transformed'],errors='ignore').corr()['SalePrice'].abs().sort_values(ascending=False).head(11).index; sns.heatmap(trans[heat_cols].corr(),annot=True,fmt='.2f',cmap='coolwarm',vmin=-1,vmax=1,linewidths=.4,cbar=False,ax=axes[4]); axes[4].set_title('5) Correlation Matrix: Top 10 Features + SalePrice')
axes[5].axis('off'); axes[5].set_title('6) GrLivArea Scaling Comparison',pad=20); mini=[axes[5].inset_axes([.05,.12,.28,.78]), axes[5].inset_axes([.37,.12,.28,.78]), axes[5].inset_axes([.69,.12,.28,.78])]; mini_data=[('Raw',X_train['GrLivArea']),('StandardScaled',scaled_data['StandardScaler'][0]['GrLivArea']),('MinMaxScaled',scaled_data['MinMaxScaler'][0]['GrLivArea'])]
for ax,(title,data) in zip(mini,mini_data): ax.hist(data,bins=30,color='#6366F1',edgecolor='white',alpha=.75); ax.set_title(title,fontsize=10); ax.tick_params(labelsize=8)
fig.suptitle('Week 3 Final Dashboard — House Price Visualization & Feature Engineering', fontsize=18, fontweight='bold', y=1.01); plt.tight_layout(); plt.savefig(OUT/'week3_dashboard.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('dashboard saved')
# fe pipeline
feature_formulas={'TotalSF':'TotalBsmtSF + 1stFlrSF + 2ndFlrSF','TotalBaths':'FullBath + .5*HalfBath + BsmtFullBath + .5*BsmtHalfBath','HouseAge':'YrSold - YearBuilt','RemodelAge':'YrSold - YearRemodAdd','HasRemodeled':'YearBuilt != YearRemodAdd','QualCond':'OverallQual * OverallCond','PricePerSF':'SalePrice / TotalSF','IsNewHouse':'YearBuilt >= YrSold - 5'}
feat_table=pd.DataFrame({'Feature':new_features,'Formula':[feature_formulas[f] for f in new_features],'Corr':[fe[f].corr(fe['SalePrice']) for f in new_features]}); transformed_skew=trans[treat_cols].skew().dropna() if len(treat_cols) else pd.Series(dtype=float); skew_compare=pd.DataFrame({'Original Skew':orig_skew.reindex(treat_cols),'Transformed Skew':transformed_skew.reindex(treat_cols)}).dropna()
fig, axes=plt.subplots(2,2,figsize=(16,11)); axes=axes.ravel(); axes[0].axis('off'); table_data=feat_table.copy(); table_data['Corr']=table_data['Corr'].map(lambda x:f'{x:.3f}'); t1=axes[0].table(cellText=table_data.values,colLabels=table_data.columns,loc='center',cellLoc='left'); t1.auto_set_font_size(False); t1.set_fontsize(8); t1.scale(1,1.6); axes[0].set_title('Panel 1 — Features Created', fontsize=14, fontweight='bold')
counts=strategy_table['strategy'].value_counts(); axes[1].bar(counts.index,counts.values,color=['#5B21B6','#0D9488','#F59E0B','#DC2626'][:len(counts)]); axes[1].set_title('Panel 2 — Encoding Decisions'); axes[1].set_ylabel('Number of Categorical Features'); axes[1].tick_params(axis='x',rotation=25)
for i,v in enumerate(counts.values): axes[1].text(i,v,str(v),ha='center',va='bottom')
if len(skew_compare):
    axes[2].scatter(skew_compare['Original Skew'],skew_compare['Transformed Skew'],alpha=.65,color='#0D9488'); ml=max(skew_compare.abs().max().max(),1); axes[2].plot([-ml,ml],[-ml,ml],color='#111827',ls='--',label='No change line'); axes[2].axhline(0,color='gray',lw=.8); axes[2].axvline(0,color='gray',lw=.8); axes[2].legend()
axes[2].set(title='Panel 3 — Skewness Treatment Before vs After', xlabel='Original Skewness', ylabel='Transformed Skewness')
axes[3].axis('off'); final_top15=final_features_info.head(15).copy(); final_top15['correlation_with_SalePrice']=final_top15['correlation_with_SalePrice'].map(lambda x:f'{x:.3f}'); t2=axes[3].table(cellText=final_top15.values,colLabels=final_top15.columns,loc='center',cellLoc='left'); t2.auto_set_font_size(False); t2.set_fontsize(8); t2.scale(1,1.55); axes[3].set_title('Panel 4 — Final Feature Set: Top 15', fontsize=14, fontweight='bold')
fig.suptitle('Week 3 Feature Engineering Pipeline Summary', fontsize=18, fontweight='bold'); plt.tight_layout(); plt.savefig(OUT/'week3_fe_pipeline.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('pipeline saved')
# written report
corr_all=fe.select_dtypes(include='number').corr()['SalePrice'].drop('SalePrice').sort_values(key=lambda s:s.abs(), ascending=False); engineered_in_top20=[c for c in corr_all.head(20).index if c in new_features]
report=f"""# Week 3 Written Analysis Report\n\n## 1) Executive Summary\nThis Week 3 project used the Kaggle House Prices dataset with 1,460 rows, 81 original columns, and `SalePrice` as the target variable. The main goal was to convert raw housing data into professional visual insights and machine-learning-ready inputs. The first key finding is that `SalePrice` is strongly right-skewed, with an original skewness of about {sale_skew:.2f}; after transformation, Box-Cox produced the lowest skewness, while log transformation also made the distribution much more normal and interpretable. The second key finding is that quality and space dominate price: `OverallQual`, `TotalSF`, and `GrLivArea` were among the strongest predictors. The third key finding is that engineered features added real value: {', '.join(engineered_in_top20)} appeared in the top correlation ranking.\n\n## 2) Visualization Insights\nThe SalePrice distribution chart revealed that house prices are not normally distributed. Most houses are concentrated in the lower-to-middle price range, while a smaller number of expensive homes creates a long right tail. The log-transformed chart showed a much more balanced shape, which is useful for regression modelling. The GrLivArea box and violin plots showed that living area also contains outliers, meaning a few houses are much larger than the majority. The multivariable scatter plot was one of the most informative charts because it showed that larger homes usually sell for more, but the strongest prices are also linked with higher `OverallQual` and larger garage capacity. The time-trend chart showed that newer construction decades generally have higher average sale prices. The neighborhood box plot showed that location matters strongly because median prices vary widely across neighborhoods. The heatmap confirmed that quality, size, basement area, garage size, and bathrooms are strongly related to price. The pair plot and FacetGrid added more statistical evidence that quality categories separate expensive homes from cheaper homes.\n\n## 3) Feature Engineering Rationale\nEight features were engineered using real-estate domain logic. `TotalSF` was created because buyers usually value total usable space, not just one individual area column. `TotalBaths` was created because full bathrooms add more value than half bathrooms, so half bathrooms were weighted by 0.5. `HouseAge` was created because older houses may require more maintenance and may sell for less unless updated. `RemodelAge` captured how recently the property was improved because a recent remodel can increase buyer interest. `HasRemodeled` turned remodeling status into a simple binary signal. `QualCond` combined overall quality and condition, because a house that is both high quality and well maintained should be more valuable. `PricePerSF` measured value density and was useful for analysis, although it should not be used as a real predictive input because it contains the target. `IsNewHouse` captured the premium that buyers may pay for recently built homes.\n\n## 4) Encoding Decisions\nThe categorical encoding strategy was chosen according to the meaning and size of each categorical column. Ordinal quality columns such as `ExterQual`, `KitchenQual`, `BsmtQual`, `GarageQual`, and `FireplaceQu` were label encoded using the ordered scale `Ex=5`, `Gd=4`, `TA=3`, `Fa=2`, `Po=1`, and `NA=0`, because these categories have a natural quality ranking. Low-cardinality nominal columns with no natural order, such as zoning, lot shape, building type, house style, foundation, and sale condition, were one-hot encoded so that the model would not assume false ordering. High-cardinality nominal columns, including `Neighborhood`, `Exterior1st`, and `Exterior2nd`, were frequency encoded to prevent excessive dummy columns. Columns with more than 50% missing values or one category dominating more than 95% were dropped because they contributed little reliable signal and could add noise.\n\n## 5) Scaling Analysis\nThree scaling methods were compared: StandardScaler, MinMaxScaler, and RobustScaler. StandardScaler centers each feature around mean 0 and standard deviation 1, making it a strong default for linear regression, ridge, lasso, SVM, and neural networks. MinMaxScaler compresses values into a fixed 0–1 range, which is helpful when bounded inputs are required, but it is sensitive to outliers. RobustScaler uses the median and interquartile range, making it better when outliers are extreme. For Week 4 linear regression, StandardScaler is the best choice because it gives all numerical features comparable scale while preserving the overall distribution structure. For tree-based models, scaling is usually not required because decision trees split using thresholds.\n\n## 6) Skewness Treatment Findings\nSkewness analysis showed that many numerical features were highly skewed, especially area, count, and one-hot encoded sparse features. In this processed dataset, {(skew.abs()>1).sum()} out of {len(skew)} numerical columns had absolute skewness above 1, which is about {(skew.abs()>1).mean()*100:.2f}%. For non-target numerical features with absolute skewness greater than 0.75 and non-negative values, log1p transformation was applied because it handles zeros and reduces right tails. For the target `SalePrice`, three transformations were compared: log1p, square root, and Box-Cox. Box-Cox produced the lowest absolute skewness, approximately {transform_summary.loc[transform_summary.transformation=='Box-Cox','skewness'].iloc[0]:.3f}, while log1p reduced skewness from about {sale_skew:.2f} to about {log_sale_skew:.2f}. Reducing skewness matters because linear models often perform better when relationships are more stable and distributions are less dominated by extreme values.\n\n## 7) Reflection\nThe hardest concept in this task was deciding how to encode categorical variables correctly because not all text columns should be treated the same way. Some categories have a real order, while others are purely names and need one-hot or frequency encoding. The most surprising pattern was the strength of engineered features. `TotalSF` achieved a very strong relationship with `SalePrice`, showing that combining related raw columns can create a clearer signal than using each column separately. Another important lesson was that a feature can be useful for analysis but unsafe for prediction; `PricePerSF` is informative, but it leaks the target. Next, I would train baseline linear regression, ridge regression, random forest, and gradient boosting models using the selected feature set and compare performance using cross-validation.\n"""
(OUT/'week3_written_report.md').write_text(report,encoding='utf-8')
readme="""# AIML Internship Week 3 — House Price Visualization & Feature Engineering\n\n## Dataset\nHouse Prices — Advanced Regression Techniques (`train.csv`)\n\n## 5 Key Findings\n1. `SalePrice` is strongly right-skewed and benefits from transformation.\n2. `OverallQual`, `TotalSF`, and `GrLivArea` are among the strongest price-related features.\n3. Newer and recently remodeled houses generally show stronger pricing patterns.\n4. Neighborhood and kitchen quality create clear price differences.\n5. Feature engineering added strong signal through `TotalSF`, `TotalBaths`, and `QualCond`.\n\n## Top 3 Engineered Features\n- `TotalSF`\n- `TotalBaths`\n- `QualCond`\n\n## Tools Used\nPython, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy\n\n## Dashboard Screenshot\n![Week 3 Dashboard](week3_dashboard.png)\n\n## Feature Engineering Pipeline\n![Feature Engineering Pipeline](week3_fe_pipeline.png)\n"""
(OUT/'README.md').write_text(readme,encoding='utf-8')
print('done')
