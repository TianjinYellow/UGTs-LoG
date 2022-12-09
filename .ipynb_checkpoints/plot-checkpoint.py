from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd  
import numpy as np

logits=np.load("./ogbn-arxiv_GCN_4000_0.03_score_only_32_z.npy")
y=np.load("./ogbn-arxiv_GCN_4000_0.03_score_only_32_y.npy")
sns.set_style("white")

mask=y<10
y=y[mask]
logits=logits[mask]

tsne=TSNE(n_components=2,verbose=1,random_state=123,perplexity=50)
z=tsne.fit_transform(logits)

df=pd.DataFrame()
length=int(z[:,0].shape[0]*1)
all_len=z[:,0].shape[0]

indexes=np.arange(all_len)
np.random.shuffle(indexes)
samples_indexes=indexes[:length]
#print("z shape",z.shape)

x_samples=z[:,0][samples_indexes]
y_samples=z[:,1][samples_indexes]
hue_samples=y[samples_indexes].tolist()
y_set=set(y[samples_indexes])
#print(len(list(y_set)))
mask=y_samples==0

df['y']=y[samples_indexes]
df['comp-1']=x_samples
df['comp-2']=y_samples




#np.save(file_path[:-4]+"_z.npy",logits[self.idx_test].cpu().detach().squeeze().numpy())
#np.save(file_path[:-4]+"_y.npy",y)
#print("x shape",x_samples.shape)
#print("y shape",y_samples.shape)
#print(len(hue_samples ))

fig=sns.scatterplot(x=x_samples, y=y_samples, hue=hue_samples,palette=sns.color_palette(n_colors= len(list(set(df['y'])))),data=df,s=100,legend=False,linewidth=0.5)
fig.set(xticklabels=[],yticklabels=[])
fig.set(xlabel=None,ylabel=None)
fig.tick_params(bottom=False,left=False,pad=0)
sns.despine(top=True, right=True, left=True, bottom=True, trim=True)
scatter=fig.get_figure()
scatter.tight_layout()
scatter.savefig("ogbn_score_32.jpg",pad_inches=0.0,dpi=600,bbox_inches='tight') 