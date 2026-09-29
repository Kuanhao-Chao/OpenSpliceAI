"""Figures for event-level distance context and matched stratum comparisons."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import deeper, derive, style
from .loading import EVENTS, SCORE_LABELS
from .figures import _column_quantiles, _save

DISTANCES = derive.SITE_DISTANCE_ORDER


def site_rates(run, out_dir, index):
    rows = deeper.site_event_table(run)
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.8))
    for event, ax in zip(SCORE_LABELS, axes):
        subset = [r for r in rows if r["event"] == event and r["threshold"] == 0.5]
        grouped = {}
        for r in subset:
            g = grouped.setdefault(r["distance"], {"n":0,"left":0,"right":0})
            g["n"] += r["n"]
            g["left"] += r["both_positive"]+r["left_only"]
            g["right"] += r["both_positive"]+r["right_only"]
        names = [d for d in DISTANCES if d in grouped]
        for side, color, marker, label in (("left",style.SPLICEAI,"o",run.left_label),
                                            ("right",style.OPENSPLICEAI,"s",run.right_label)):
            values = [grouped[d][side]/grouped[d]["n"] for d in names]
            ax.plot(names, [x if x else np.nan for x in values], marker=marker, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)), names)
        ax.set_xlim(-0.3, len(names)-0.7)
        ax.set_title(style.EVENT_TITLES[event])
        ax.tick_params(axis="x", rotation=50)
        ax.set_xlabel("distance to an internal boundary (bp)")
        style.clean(ax)
    axes[0].set_ylabel("fraction with score ≥ 0.5")
    axes[0].legend(fontsize=7)
    _save(fig,out_dir,"f12_site_distance.png",index,
          "Event-specific call rates at threshold 0.5, pooled over nearest-site types. "
          "Zeros are omitted from the logarithmic axis and retained in the accompanying table. "
          "Distance is annotation context, not an experimental truth label.")


def distributions(run, out_dir, index):
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.0), sharex=True)
    x = np.arange(run.score_bins)/run.score_bins
    for row, (distances, title) in enumerate(((('at_site','1-2'),'At or within 2 bp'),
                                            (('>500',),'More than 500 bp away'))):
        for col, event in enumerate(SCORE_LABELS):
            ax = axes[row,col]
            left,right,_ = deeper.histograms(run,distances,event)
            for values,color,label,linestyle in ((left,style.SPLICEAI,run.left_label,"-"),
                                                (right,style.OPENSPLICEAI,run.right_label,"--")):
                if values.sum():
                    survival = np.cumsum(values[::-1])[::-1]/values.sum()
                    ax.plot(x,np.where(survival>0,survival,np.nan),color=color,
                            linestyle=linestyle,label=label)
            ax.set_yscale("log")
            ax.set_xlim(0,1)
            ax.set_title(style.EVENT_TITLES[event])
            style.clean(ax)
            if row==1:
                ax.set_xlabel("score cutoff")
        axes[row,0].set_ylabel(title+"\nfraction at or above cutoff")
    axes[0,0].legend(fontsize=7)
    _save(fig,out_dir,"f13_site_score_distributions.png",index,
          "Score survival distributions near internal splice boundaries versus variants more than 500 bp away, "
          "for each event. Curves use histogram edges at 0.005 resolution. The distant group includes "
          "any genomic context present in the input; it is not labelled deep intron.")


def conditioned_calibration(run,out_dir,index):
    fig, axes = plt.subplots(2,5,figsize=(14.5,5.7),sharex=True,sharey=True)
    x=(np.arange(run.score_bins)+0.5)/run.score_bins
    for row,(distances,title) in enumerate(((('at_site','1-2'),'At or within 2 bp'),
                                           (('>500',),'More than 500 bp away'))):
        for col,event in enumerate(SCORE_LABELS):
            ax=axes[row,col]
            _,_,joint=deeper.histograms(run,distances,event)
            q=_column_quantiles(joint,run.score_bins,(0.25,0.5,0.75))
            ax.vlines(x,q[0.25],q[0.75],color=style.SPLICEAI,alpha=.35,linewidth=2)
            ax.plot(x,q[0.5],color=style.OPENSPLICEAI,marker="o",markersize=2,linewidth=1)
            ax.plot([0,1],[0,1],color=style.INK_MUTED,ls="--",lw=.8)
            ax.set_title(style.EVENT_TITLES[event])
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            style.clean(ax,"both")
            if row==1:
                ax.set_xlabel(run.left_label+" score")
        axes[row,0].set_ylabel(title+"\n"+run.right_label+" score")
    _save(fig,out_dir,"f14_site_conditioned_calibration.png",index,
          "Median and interquartile range of the right score conditional on the left score, separated "
          "by annotation proximity. Empty conditioning bins remain gaps. This is agreement calibration "
          "against a comparator, not calibration against observed splice outcomes.")


def conditioned_dp(run,out_dir,index):
    rows=[r for r in deeper.site_dp_table(run) if r["threshold"]==0.5]
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),sharey=True)
    for event,ax in zip(EVENTS,axes):
        for tolerance,marker,color,line in ((0,"o",style.SPLICEAI,"-"),
                                            (2,"s",style.OPENSPLICEAI,"--")):
            by={}
            for r in rows:
                if r["event"]!=event or r["tolerance_bp"]!=tolerance:
                    continue
                n,within=by.get(r["distance"],(0,0))
                by[r["distance"]]=(n+r["eligible"],within+r["within"])
            names=[d for d in DISTANCES if d in by]
            rates=[by[d][1]/by[d][0] if by[d][0] else np.nan for d in names]
            ax.plot(names,rates,marker=marker,color=color,linestyle=line,
                    label=f"within {tolerance} bp")
        ax.tick_params(axis="x",rotation=50)
        ax.set_title(style.EVENT_TITLES[event])
        ax.set_ylim(0,1.02)
        style.clean(ax)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("position agreement when both scores ≥ 0.5")
    _save(fig,out_dir,"f15_site_conditioned_dp.png",index,
          "Predicted-position agreement by variant proximity. Both event scores must be at least 0.5; "
          "DP=0 is eligible. Empty eligible groups are gaps, and the table retains all five tolerances.")


def seed_strata(seed,models,out_dir,index):
    rows=deeper.matched_strata(seed,models)
    if not rows:
        return
    fig,axes=plt.subplots(1,4,figsize=(13,3.6),sharey=True)
    for event,ax in zip(EVENTS,axes):
        for model,color,line in zip(models,(style.SPLICEAI,style.OPENSPLICEAI),("-","--")):
            values=sorted(r["mae_ratio"] for r in rows if r["dimension"]=="gene" and
                          r["event"]==event and r["model_arm"]==model.arm and
                          r["n"]>=1000 and r["mae_ratio"] is not None and r["mae_ratio"]>0)
            if values:
                ax.plot(values,np.arange(1,len(values)+1)/len(values),color=color,
                        linestyle=line,label=model.right_label)
        ax.axvline(1,color=style.INK_MUTED,ls="--",lw=1)
        ax.set_xscale("log")
        ax.set_title(style.EVENT_TITLES[event])
        ax.set_xlabel("method MAE / seed MAE")
        style.clean(ax)
    axes[0].set_ylabel("fraction of eligible genes")
    axes[0].legend(title="SpliceAI vs",fontsize=8)
    _save(fig,out_dir,"f16_seed_method_by_gene.png",index,
          "Distribution across genes of method-to-seed MAE ratios on the exact three-way intersection, "
          "restricted in the figure to genes with at least 1,000 paired observations and positive, defined "
          "ratios. The complete table retains excluded and undefined entries and all genomic strata.")


def render(run,seed,models,out_dir,index):
    if not run.raw.get("depth"):
        return
    site_rates(run,out_dir,index)
    distributions(run,out_dir,index)
    conditioned_calibration(run,out_dir,index)
    conditioned_dp(run,out_dir,index)
    if seed is not None:
        seed_strata(seed,models,out_dir,index)
