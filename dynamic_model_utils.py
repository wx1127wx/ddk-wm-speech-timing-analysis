"""Minimal local model utilities used by Experiment 03."""
from __future__ import annotations

import math
from numpy.polynomial.hermite import hermgauss
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from scipy.special import expit, logsumexp


def bh_adjust(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    adjusted[order] = np.minimum(1, np.minimum.accumulate((pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1))[::-1])[::-1])
    return adjusted


class DiagonalLMM:
    def __init__(self, y, X, Z, groups, fixed_names, random_names, reml=True):
        self.y, self.X, self.Z = map(lambda a: np.asarray(a, float), (y, X, Z))
        self.groups = np.asarray(groups); self.fixed_names = list(fixed_names); self.random_names = list(random_names)
        self.unique_groups, inverse = np.unique(self.groups, return_inverse=True)
        self.blocks = [np.flatnonzero(inverse == i) for i in range(len(self.unique_groups))]
        self.n, self.p, self.q = len(self.y), self.X.shape[1], self.Z.shape[1]
        self.reml = bool(reml)

    def _profile(self, theta, return_parts=False):
        variances = np.exp(2*np.asarray(theta)); random_var, residual_var = variances[:self.q], variances[-1]
        xtvix = np.zeros((self.p,self.p)); xtviy = np.zeros(self.p); logdet_v = 0.; cache=[]
        for idx in self.blocks:
            yi, xi, zi = self.y[idx], self.X[idx], self.Z[idx]
            vi = (zi*random_var) @ zi.T; vi.flat[::vi.shape[0]+1] += residual_var
            try: cf = linalg.cho_factor(vi, lower=True, check_finite=False)
            except linalg.LinAlgError: return (1e100,None) if return_parts else 1e100
            xtvix += xi.T @ linalg.cho_solve(cf, xi, check_finite=False); xtviy += xi.T @ linalg.cho_solve(cf, yi, check_finite=False)
            logdet_v += 2*np.log(np.diag(cf[0])).sum(); cache.append((idx,cf))
        try:
            beta = linalg.solve(xtvix, xtviy, assume_a="sym"); sign, logdet_x = np.linalg.slogdet(xtvix)
            if sign <= 0: raise linalg.LinAlgError
        except linalg.LinAlgError: return (1e100,None) if return_parts else 1e100
        quad=0.
        for idx,cf in cache:
            r=self.y[idx]-self.X[idx]@beta; quad += r @ linalg.cho_solve(cf,r,check_finite=False)
        if self.reml:
            nll=.5*(logdet_v+logdet_x+quad+(self.n-self.p)*np.log(2*np.pi))
        else:
            nll=.5*(logdet_v+quad+self.n*np.log(2*np.pi))
        return (nll,(beta,linalg.inv(xtvix),random_var,residual_var,cache)) if return_parts else nll

    def fit(self):
        beta0=np.linalg.lstsq(self.X,self.y,rcond=None)[0]; sd=max(np.std(self.y-self.X@beta0),1e-3)
        starts=[np.log(np.r_[np.repeat(sd*.7,self.q),sd*.7]),np.log(np.r_[np.repeat(sd*.25,self.q),sd]),np.log(np.r_[np.repeat(sd,self.q),sd*.5])]
        fits=[optimize.minimize(self._profile,s,method="L-BFGS-B",bounds=[(np.log(max(sd*1e-4,1e-10)),np.log(sd*100))]*(self.q+1),options={"maxiter":1200,"ftol":1e-11,"gtol":1e-7}) for s in starts]
        best=min(fits,key=lambda x:x.fun); nll,parts=self._profile(best.x,return_parts=True); beta,cov,rv,resv,cache=parts; se=np.sqrt(np.maximum(np.diag(cov),0)); z=beta/se
        likelihood_key = "nll_reml" if self.reml else "nll_ml"
        self.result={"optimizer":best,likelihood_key:nll,"nll":nll,"beta":beta,"beta_cov":cov,"se":se,"z":z,"p":2*stats.norm.sf(abs(z)),"random_sd":np.sqrt(rv),"residual_sd":math.sqrt(resv),"ranef":pd.DataFrame()}
        return self.result


class LogisticRandomInterceptSlope:
    def __init__(self,y,X,difficulty,groups,gh_points=11):
        self.y=np.asarray(y,float); self.X=np.asarray(X,float); self.d=np.asarray(difficulty,float); self.groups=np.asarray(groups)
        self.unique_groups,inv=np.unique(self.groups,return_inverse=True); self.blocks=[np.flatnonzero(inv==i) for i in range(len(self.unique_groups))]
        nodes,weights=hermgauss(gh_points); n0,n1=np.meshgrid(nodes,nodes,indexing="ij"); w0,w1=np.meshgrid(weights,weights,indexing="ij")
        self.nodes0=n0.ravel(); self.nodes1=n1.ravel(); self.logweights=np.log((w0*w1).ravel())-np.log(np.pi)

    def negloglik(self,params):
        beta=params[:self.X.shape[1]]; sd0,sd1=np.exp(params[-2:]); b0=np.sqrt(2)*sd0*self.nodes0; b1=np.sqrt(2)*sd1*self.nodes1; total=0.
        for idx in self.blocks:
            eta=(self.X[idx]@beta)[None,:]+b0[:,None]+b1[:,None]*self.d[idx][None,:]
            ll=(self.y[idx][None,:]*eta-np.logaddexp(0.,eta)).sum(axis=1); total += logsumexp(self.logweights+ll)
        return -total

    def fit(self):
        p_low=np.mean(self.y[self.d<0]); p_high=np.mean(self.y[self.d>0]); logit=lambda p:np.log(p/(1-p)); start=np.zeros(self.X.shape[1]); start[0]=(logit(p_low)+logit(p_high))/2; start[1]=logit(p_high)-logit(p_low)
        starts=[np.r_[start,np.log([.8,.5])],np.r_[start,np.log([1.2,.2])],np.r_[start,np.log([.5,1.])]]; bounds=[(-12,12)]*len(start)+[(-5,2),(-5,2)]
        fits=[optimize.minimize(self.negloglik,s,method="L-BFGS-B",bounds=bounds,options={"maxiter":1000,"ftol":1e-10,"gtol":1e-6}) for s in starts]; best=min(fits,key=lambda x:x.fun); x=best.x; k=len(x); h=np.maximum(2e-4,abs(x)*2e-4); H=np.zeros((k,k)); f0=self.negloglik(x)
        for i in range(k):
            ei=np.zeros(k); ei[i]=h[i]; H[i,i]=(self.negloglik(x+ei)-2*f0+self.negloglik(x-ei))/h[i]**2
            for j in range(i+1,k):
                ej=np.zeros(k); ej[j]=h[j]; H[i,j]=H[j,i]=(self.negloglik(x+ei+ej)-self.negloglik(x+ei-ej)-self.negloglik(x-ei+ej)+self.negloglik(x-ei-ej))/(4*h[i]*h[j])
        cov=np.linalg.pinv(H); beta=x[:self.X.shape[1]]; se=np.sqrt(np.maximum(np.diag(cov)[:self.X.shape[1]],0)); z=beta/se
        self.result={"optimizer":best,"beta":beta,"se":se,"z":z,"p":2*stats.norm.sf(abs(z)),"cov":cov,"random_sd":np.exp(x[-2:])}; return self.result

    def conditional_probability(self,difficulty,progress,result=None):
        result=result or self.result; x=np.array([1.,difficulty,progress,difficulty*progress]); eta=x@result["beta"]; se=math.sqrt(x@result["cov"][:4,:4]@x); return expit(eta),expit(eta-1.96*se),expit(eta+1.96*se)
