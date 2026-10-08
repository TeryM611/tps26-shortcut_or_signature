#!/usr/bin/env python3
"""
Self-contained ML utilities in NumPy (no sklearn/xgboost).
 - rank-based AUC (Mann-Whitney U)
 - histogram-based Random Forest classifier
 - L2 logistic regression (Adam)
 - grouped stratified K-fold CV
 - permutation feature importance
"""
import numpy as np

# ----------------------------- metrics --------------------------------------
def auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n1 = int(y_true.sum()); n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(score)+1, dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j+1 < len(s_sorted) and s_sorted[j+1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j+1] = (i+1 + j+1)/2.0
        i = j+1
    ranks[order] = ranks_sorted
    sum_pos = ranks[y_true == 1].sum()
    u = sum_pos - n1*(n1+1)/2.0
    return u/(n1*n0)

def binary_metrics(y_true, score, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum()); fp = int(((pred==1)&(y_true==0)).sum())
    tn = int(((pred==0)&(y_true==0)).sum()); fn = int(((pred==0)&(y_true==1)).sum())
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec = tp/(tp+fn) if tp+fn>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    fpr = fp/(fp+tn) if fp+tn>0 else 0.0
    acc = (tp+tn)/len(y_true)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=prec,recall=rec,f1=f1,fpr=fpr,accuracy=acc)

# ------------------------- histogram random forest --------------------------
class _HistTree:
    def __init__(self, max_depth=10, min_leaf=20, max_features=None, rng=None):
        self.max_depth=max_depth; self.min_leaf=min_leaf
        self.max_features=max_features; self.rng=rng or np.random.default_rng()
        self.nodes=[]  # each: dict

    def fit(self, Xb, y, nbins):
        self.nbins=nbins
        self.n_features=Xb.shape[1]
        idx=np.arange(len(y))
        self.nodes=[]
        self._build(Xb, y, idx, 0)
        self._arrayize()
        return self

    def _arrayize(self):
        n=len(self.nodes)
        self.a_leaf=np.zeros(n,dtype=bool); self.a_feat=np.zeros(n,dtype=np.int32)
        self.a_thr=np.zeros(n,dtype=np.int32); self.a_left=np.zeros(n,dtype=np.int32)
        self.a_right=np.zeros(n,dtype=np.int32); self.a_val=np.zeros(n,dtype=np.float64)
        for i,nd in enumerate(self.nodes):
            self.a_val[i]=nd['p']
            if nd['leaf']:
                self.a_leaf[i]=True
            else:
                self.a_feat[i]=nd['f']; self.a_thr[i]=nd['thr']
                self.a_left[i]=nd['l']; self.a_right[i]=nd['r']

    def predict_proba_arr(self, Xb):
        n=len(Xb); node=np.zeros(n,dtype=np.int32)
        for _ in range(self.max_depth+3):
            active=~self.a_leaf[node]
            if not active.any(): break
            nd=node[active]
            fa=self.a_feat[nd]
            xv=Xb[active, fa]
            go_left=xv<=self.a_thr[nd]
            node[active]=np.where(go_left, self.a_left[nd], self.a_right[nd])
        return self.a_val[node]

    def _build(self, Xb, y, idx, depth):
        node_id=len(self.nodes)
        self.nodes.append(None)
        pos=y[idx].sum(); n=len(idx)
        p = pos/n if n>0 else 0.0
        make_leaf = (depth>=self.max_depth or n<2*self.min_leaf or pos==0 or pos==n)
        best=None
        if not make_leaf:
            mf = self.max_features or self.n_features
            feats = self.rng.choice(self.n_features, size=min(mf,self.n_features), replace=False)
            yi = y[idx].astype(np.float64)
            best_gain=1e-12; best_feat=-1; best_thr=-1
            parent_imp = 1.0 - p*p - (1-p)*(1-p)
            for f in feats:
                xb = Xb[idx, f]
                cnt = np.bincount(xb, minlength=self.nbins).astype(np.float64)
                pcnt = np.bincount(xb, weights=yi, minlength=self.nbins)
                ctot = cnt.sum(); ptot = pcnt.sum()
                cleft = np.cumsum(cnt); pleft = np.cumsum(pcnt)
                cright = ctot - cleft; pright = ptot - pleft
                valid = (cleft>=self.min_leaf)&(cright>=self.min_leaf)
                if not valid.any(): continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    pl = np.where(cleft>0, pleft/cleft, 0.0)
                    pr = np.where(cright>0, pright/cright, 0.0)
                    gl = 1.0 - pl*pl - (1-pl)*(1-pl)
                    gr = 1.0 - pr*pr - (1-pr)*(1-pr)
                    child = (cleft*gl + cright*gr)/ctot
                gain = parent_imp - child
                gain[~valid] = -1
                b = np.argmax(gain[:-1]) if len(gain)>1 else 0
                if gain[b] > best_gain:
                    best_gain=gain[b]; best_feat=int(f); best_thr=int(b)
            if best_feat>=0:
                best=(best_feat,best_thr)
        if best is None:
            self.nodes[node_id]={'leaf':True,'p':p}
            return node_id
        f,thr=best
        xb=Xb[idx,f]
        left_mask = xb<=thr
        left_idx=idx[left_mask]; right_idx=idx[~left_mask]
        if len(left_idx)<self.min_leaf or len(right_idx)<self.min_leaf:
            self.nodes[node_id]={'leaf':True,'p':p}; return node_id
        l=self._build(Xb,y,left_idx,depth+1)
        r=self._build(Xb,y,right_idx,depth+1)
        self.nodes[node_id]={'leaf':False,'f':f,'thr':thr,'l':l,'r':r,'p':p}
        return node_id

    def predict_proba(self, Xb):
        out=np.empty(len(Xb))
        for i in range(len(Xb)):
            nid=0
            while True:
                nd=self.nodes[nid]
                if nd['leaf']: out[i]=nd['p']; break
                if Xb[i,nd['f']]<=nd['thr']: nid=nd['l']
                else: nid=nd['r']
        return out

    def predict_proba_fast(self, Xb):
        # vectorized traversal
        n=len(Xb); out=np.zeros(n); active=np.arange(n); cur=np.zeros(n,dtype=int)
        for _ in range(self.max_depth+2):
            leaves=np.array([self.nodes[c]['leaf'] for c in cur])
            done=leaves
            for k in np.where(done)[0]:
                out[k]=self.nodes[cur[k]]['p']
            todo=np.where(~done)[0]
            if len(todo)==0: break
            for k in todo:
                nd=self.nodes[cur[k]]
                cur[k]= nd['l'] if Xb[k,nd['f']]<=nd['thr'] else nd['r']
        # finalize any leftover
        for k in range(n):
            nd=self.nodes[cur[k]]
            if nd['leaf']: out[k]=nd['p']
        return out

class HistRandomForest:
    def __init__(self, n_trees=60, max_depth=12, min_leaf=20, max_features='sqrt',
                 nbins=64, subsample=1.0, seed=0):
        self.n_trees=n_trees; self.max_depth=max_depth; self.min_leaf=min_leaf
        self.max_features=max_features; self.nbins=nbins; self.subsample=subsample
        self.seed=seed

    def _bin(self, X, fit=False):
        if fit:
            self.edges=[]
            qs=np.linspace(0,1,self.nbins+1)[1:-1]
            for j in range(X.shape[1]):
                col=X[:,j]
                e=np.quantile(col, qs)
                e=np.unique(e)
                if len(e)==0: e=np.array([col.mean()])
                self.edges.append(e)
        Xb=np.empty(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            Xb[:,j]=np.searchsorted(self.edges[j], X[:,j], side='right')
        self.nbins_eff=max(len(e)+1 for e in self.edges)
        return Xb

    def fit(self, X, y):
        rng=np.random.default_rng(self.seed)
        Xb=self._bin(X, fit=True)
        y=np.asarray(y).astype(int)
        n=len(y)
        mf = int(np.sqrt(X.shape[1])) if self.max_features=='sqrt' else (self.max_features or X.shape[1])
        self.trees=[]
        for t in range(self.n_trees):
            tr_rng=np.random.default_rng(self.seed*1000+t)
            m=int(n*self.subsample) if self.subsample<1.0 else n
            boot=tr_rng.integers(0,n,size=m)
            tree=_HistTree(self.max_depth,self.min_leaf,mf,tr_rng)
            tree.fit(Xb[boot], y[boot], self.nbins_eff)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        Xb=self._bin(X, fit=False)
        s=np.zeros(len(X))
        for tree in self.trees:
            s+=tree.predict_proba_arr(Xb)
        return s/len(self.trees)

# ---------------------------- logistic regression ---------------------------
class LogisticReg:
    def __init__(self, l2=1e-3, lr=0.05, epochs=300, seed=0):
        self.l2=l2; self.lr=lr; self.epochs=epochs; self.seed=seed
    def fit(self, X, y):
        self.mu=X.mean(0); self.sd=X.std(0); self.sd[self.sd==0]=1
        Xs=(X-self.mu)/self.sd
        n,d=Xs.shape
        w=np.zeros(d); b=0.0
        mw=np.zeros(d); vw=np.zeros(d); mb=0.0; vb=0.0
        b1,b2,eps=0.9,0.999,1e-8
        y=np.asarray(y,dtype=float)
        for t in range(1,self.epochs+1):
            z=Xs@w+b; p=1/(1+np.exp(-z))
            gw=Xs.T@(p-y)/n + self.l2*w; gb=(p-y).mean()
            mw=b1*mw+(1-b1)*gw; vw=b2*vw+(1-b2)*gw*gw
            mb=b1*mb+(1-b1)*gb; vb=b2*vb+(1-b2)*gb*gb
            mwh=mw/(1-b1**t); vwh=vw/(1-b2**t); mbh=mb/(1-b1**t); vbh=vb/(1-b2**t)
            w-=self.lr*mwh/(np.sqrt(vwh)+eps); b-=self.lr*mbh/(np.sqrt(vbh)+eps)
        self.w=w; self.b=b; return self
    def predict_proba(self, X):
        Xs=(X-self.mu)/self.sd
        return 1/(1+np.exp(-(Xs@self.w+self.b)))

# ---------------------------- grouped CV ------------------------------------
def grouped_kfold_indices(groups, k=5, seed=0):
    groups=np.asarray(groups)
    uniq=np.unique(groups)
    rng=np.random.default_rng(seed); rng.shuffle(uniq)
    folds=[uniq[i::k] for i in range(k)]
    for i in range(k):
        test_g=set(folds[i].tolist())
        test_mask=np.array([g in test_g for g in groups])
        yield np.where(~test_mask)[0], np.where(test_mask)[0]

def stratified_kfold_indices(y, k=5, seed=0):
    y=np.asarray(y); rng=np.random.default_rng(seed)
    idx0=np.where(y==0)[0]; idx1=np.where(y==1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    folds0=[idx0[i::k] for i in range(k)]; folds1=[idx1[i::k] for i in range(k)]
    for i in range(k):
        test=np.concatenate([folds0[i],folds1[i]])
        train=np.concatenate([folds0[j] for j in range(k) if j!=i]+[folds1[j] for j in range(k) if j!=i])
        yield train, test

def cv_auc(model_factory, X, y, splits):
    aucs=[]
    for tr,te in splits:
        m=model_factory(); m.fit(X[tr],y[tr])
        s=m.predict_proba(X[te]); aucs.append(auc(y[te],s))
    return np.array(aucs)

# ------------------------ permutation importance ----------------------------
def permutation_importance(model, X, y, feat_names, n_repeats=5, seed=0):
    rng=np.random.default_rng(seed)
    base=auc(y, model.predict_proba(X))
    imps=np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops=[]
        for _ in range(n_repeats):
            Xp=X.copy(); Xp[:,j]=rng.permutation(Xp[:,j])
            drops.append(base-auc(y, model.predict_proba(Xp)))
        imps[j]=np.mean(drops)
    order=np.argsort(-imps)
    return [(feat_names[i], float(imps[i])) for i in order], base
