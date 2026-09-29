"""Supplementary displays derived from frozen counters and exported tables."""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .figures import _column_quantiles
from .loading import EVENTS, SCORE_LABELS

DISTANCES = ("at_site", "1-2", "3-10", "11-50", "51-500", ">500")
LABELS = ("At site", "1–2", "3–10", "11–50", "51–500", ">500")


def read_table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_table(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def position_groups(rows, event, tolerance):
    groups = {d: [0, 0] for d in DISTANCES}
    for r in rows:
        if r['event'] == event and float(r['threshold']) == .5 and int(r['tolerance_bp']) == tolerance:
            groups[r['distance']][0] += int(r['eligible'])
            groups[r['distance']][1] += int(r['within'])
    result = []
    for d, (n, within) in groups.items():
        if not 0 <= within <= n:
            raise ValueError('invalid position counts')
        result.append(dict(event=event, distance=d, tolerance_bp=tolerance, eligible=n,
                           within=within, mismatches=n-within, rate=within/n if n else None))
    return result


def render(facts, arrays, study, out, save):
    directory, data = out/'figures', out/'figure_data'
    data.mkdir(exist_ok=True)
    primary = facts['primary']
    tables = study/'publication/tables/A_rs10_genomewide'
    # S1: distinct counting units are separate rows, not a misleading funnel.
    coverage = primary['coverage']
    entries = [('Source VCF rows', coverage['source_rows']),
               ('Distinct variant groups', coverage['variant_groups']),
               ('Paired variant–gene annotations', coverage['paired_annotations']),
               ('SpliceAI-only annotations', coverage['left_only_annotations']),
               ('OpenSpliceAI-only annotations', coverage['right_only_annotations'])]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axis('off')
    ax.set_title('Primary comparison: counting units and pairing coverage', loc='left')
    for i, (label, n) in enumerate(entries):
        y = .88-i*.17
        ax.text(.02, y, label, fontsize=11)
        ax.text(.97, y, f'{n:,}', ha='right', fontsize=11, weight='bold')
    save(fig, directory, 'supp01_coverage')
    write_table(data/'coverage_units.csv', [dict(unit=k, count=v) for k, v in entries])
    # S4: the residual-error share is not a proportion of variants.
    rows = [primary['quantization'][e] for e in SCORE_LABELS]
    xs = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for offset, key, label, color in [(-.19,'raw_exact_match_rate','Reported scores',style.SPLICEAI),
                                     (.19,'rounded_exact_match_rate','OpenSpliceAI rounded to 2 decimals',style.OPENSPLICEAI)]:
        axes[0].bar(xs+offset, [r[key] for r in rows], width=.36, label=label, color=color)
    axes[0].set(title='Exact score agreement', ylabel='Fraction of paired annotations', ylim=(0, 1))
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5,-.15), fontsize=8)
    values = [r['quantization_adjusted_mae']/r['raw_mae'] for r in rows]
    axes[1].bar(xs, values, color=style.NEUTRAL_FILL)
    for x, y in zip(xs, values):
        axes[1].text(x, y+.015, f'{y:.3f}', ha='center', fontsize=8)
    axes[1].set(title='Residual error after a 0.005 allowance', ylabel='Residual MAE / original MAE', ylim=(0,1))
    for ax in axes:
        ax.set_xticks(xs, SCORE_LABELS)
        style.clean(ax)
    save(fig, directory, 'supp04_precision')
    # S5: exact counts, independently normalized.
    rows = primary['dominant']['matrix']
    labels = [r['left_dominant'] for r in rows]
    counts = np.asarray([[r['right_'+e] for e in labels] for r in rows], dtype=np.int64)
    denominators = counts.sum(axis=1)
    matrix = counts/denominators[:,None]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    im = ax.imshow(matrix, cmap=style.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(6), labels)
    ax.set_yticks(range(6), [f'{e}  (n={n:,})' for e,n in zip(labels,denominators)])
    ax.set(xlabel='OpenSpliceAI dominant event', ylabel='SpliceAI dominant event', title='Dominant-event labels: row proportions')
    exported = []
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            v = matrix[i,j]
            exported.append(dict(spliceai=left, openspliceai=right, count=int(counts[i,j]), denominator=int(denominators[i]), rate=float(v)))
            if v >= .005:
                ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color=style.SURFACE if v>.55 else style.INK)
    ax.grid(False)
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label='Fraction of SpliceAI row')
    save(fig,directory,'supp05_dominant_event')
    write_table(data/'dominant_row_proportions.csv',exported)
    # S6 and S7 share the exact same near/far domains and histogram cache.
    bins = int(arrays['bins'])
    for conditional, name in [(False,'supp06_site_distributions'),(True,'supp07_site_score_agreement')]:
        fig, axes = plt.subplots(2,5,figsize=(15,6.5),sharex=True,sharey=True)
        for row,(cohort,title) in enumerate([('near','At / within 2 bp'),('far','More than 500 bp')]):
            for col,e in enumerate(SCORE_LABELS):
                ax = axes[row,col]
                joint = arrays[cohort+'_'+e]
                n = int(joint.sum())
                if conditional:
                    x = (np.arange(bins)+.5)/bins
                    q = _column_quantiles(joint,bins,(.25,.5,.75))
                    ax.vlines(x,q[.25],q[.75],color=style.SPLICEAI,alpha=.35,lw=2)
                    ax.plot(x,q[.5],'.',color=style.OPENSPLICEAI,ms=3)
                    ax.plot([0,1],[0,1],'--',color=style.INK_MUTED,lw=.8)
                    ax.set(ylim=(0,1))
                else:
                    x = np.arange(bins)/bins
                    for side,color,line,label in [('left',style.SPLICEAI,'-','SpliceAI'),('right',style.OPENSPLICEAI,'--','OpenSpliceAI')]:
                        counts = arrays[cohort+'_'+side+'_'+e]
                        survival = counts[::-1].cumsum()[::-1]/n
                        ax.plot(x,np.where(survival>0,survival,np.nan),color=color,ls=line,label=label,marker='.',ms=2)
                    ax.set(yscale='log',ylim=(1e-10,1.1))
                ax.set(xlim=(0,1),title=f'{e}: {style.EVENT_TITLES[e]}')
                if row == 1:
                    ax.set_xlabel('SpliceAI score' if conditional else 'Score cutoff')
                if col == 0:
                    metric = 'OpenSpliceAI score' if conditional else 'Fraction at / above cutoff'
                    ax.set_ylabel(f'{title}\nn={n:,}\n{metric}')
                style.clean(ax)
        if not conditional:
            axes[0,0].legend(fontsize=8)
        save(fig,directory,name)
    # S8: absent eligible populations keep their common x position.
    rows = read_table(tables/'site_dp.csv')
    fig, axes = plt.subplots(2,2,figsize=(9,7.5),sharey=True)
    exported = []
    for e,ax in zip(EVENTS,axes.flat):
        for tolerance,marker,color,line in [(0,'o',style.SPLICEAI,'-'),(2,'s',style.OPENSPLICEAI,'--')]:
            groups = position_groups(rows,e,tolerance)
            exported.extend(groups)
            ax.plot(range(6),[r['rate'] if r['rate'] is not None else np.nan for r in groups],marker=marker,color=color,ls=line,label='Exact' if tolerance==0 else 'Within 2 bp')
            if tolerance == 0:
                for x,r in enumerate(groups):
                    label = f"n={r['eligible']:,}\nmiss={r['mismatches']:,}" if r['eligible'] else 'No joint\ncalls'
                    ax.text(x,.03,label,rotation=90,ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(6),LABELS,rotation=45,ha='right')
        ax.set(title=style.EVENT_TITLES[e],ylim=(0,1.04),xlim=(-.4,5.4),xlabel='Variant distance (bp)')
        style.clean(ax)
    for ax in axes[:,0]:
        ax.set_ylabel('Position agreement: both scores ≥ 0.5')
    axes[0,0].legend(loc='center',fontsize=8,ncol=2)
    save(fig,directory,'supp08_site_positions')
    write_table(data/'position_context.csv',exported)
    # S9: identical log scales and explicit eligible gene counts.
    rows = read_table(study/'publication/tables/seed_versus_method_strata.csv')
    fig,axes = plt.subplots(1,4,figsize=(14,4.4),sharey=True,sharex=True)
    stats = []
    for e,ax in zip(EVENTS,axes):
        for arm,seed,color,line in [('C_rs10_matched','rs10',style.SPLICEAI,'-'),('D_rs13_matched','rs13',style.OPENSPLICEAI,'--')]:
            eligible = [r for r in rows if r['dimension']=='gene' and r['event']==e and r['model_arm']==arm and int(r['n'])>=1000]
            values = sorted(float(r['mae_ratio']) for r in eligible if r['mae_ratio'] and float(r['mae_ratio'])>0)
            ax.plot(values,np.arange(1,len(values)+1)/len(values),color=color,ls=line,label=f'{seed}: n={len(values):,}')
            stats.append(dict(event=e,arm=arm,eligible_before_ratio_filter=len(eligible),plotted=len(values),excluded=len(eligible)-len(values),median=float(np.median(values)),fraction_above_one=float(np.mean(np.array(values)>1))))
        ax.axvline(1,color=style.INK_MUTED,ls='--',lw=1)
        ax.set(xscale='log',xlim=(.01,1e5),ylim=(0,1),title=style.EVENT_TITLES[e],xlabel='Method MAE / seed MAE')
        ax.legend(loc='lower right',fontsize=7)
        style.clean(ax)
    axes[0].set_ylabel('Fraction of plotted genes')
    save(fig,directory,'supp09_seed_by_gene')
    write_table(data/'seed_gene_summary.csv',stats)
    # S11: retain the descriptive scatter, identify MAX and its algebraic boundary.
    rows = [r for r in read_table(tables/'stratum_gene.csv') if int(r['n'])>=1000]
    means,biases,counts = [np.asarray([float(r[k]) for r in rows]) for k in ('mean_spliceai','bias','n')]
    fig,axes = plt.subplots(1,2,figsize=(12,4.3))
    sizes = 3+22*(counts-counts.min())/max(1,counts.max()-counts.min())
    axes[0].scatter(means,biases,s=sizes,color=style.SPLICEAI,alpha=.35,lw=0)
    axes[0].axhline(0,color=style.INK_MUTED,lw=1)
    axes[0].plot([0,float(means.max())],[0,-float(means.max())],':',color=style.INK_MUTED,lw=.8,label='Lower bound: mean OpenSpliceAI = 0')
    selected=[]
    for r in sorted(rows,key=lambda r:float(r['bias'])):
        x,y=float(r['mean_spliceai']),float(r['bias'])
        if any(abs(x-px)/np.ptp(means)<.06 and abs(y-py)/np.ptp(biases)<.05 for px,py in selected):
            continue
        selected.append((x,y))
        axes[0].annotate(r['stratum'],(x,y),xytext=(4,-1),textcoords='offset points',fontsize=6.5)
        if len(selected)==12:
            break
    axes[0].set(xlabel='Gene mean SpliceAI MAX',ylabel='Gene mean MAX: OpenSpliceAI − SpliceAI',title=f'{len(rows):,} genes with ≥1,000 paired annotations')
    axes[0].legend(loc='lower left',fontsize=7)
    axes[1].hist(biases,bins=60,color=style.NEUTRAL_FILL)
    axes[1].axvline(0,color=style.INK_MUTED,lw=1)
    axes[1].axvline(float(np.median(biases)),color=style.OPENSPLICEAI,label=f'Median = {np.median(biases):.4f}')
    axes[1].set(xlabel='Gene mean MAX: OpenSpliceAI − SpliceAI',ylabel='Number of genes',title='Distribution of gene-level mean differences')
    axes[1].legend(fontsize=8)
    for ax in axes:
        style.clean(ax)
    save(fig,directory,'supp11_gene_divergence')
    write_table(data/'gene_summary.csv',[dict(genes=len(rows),negative=int((biases<0).sum()),median_bias=float(np.median(biases)))])
