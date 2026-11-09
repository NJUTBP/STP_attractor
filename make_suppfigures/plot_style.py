"""Fixed-width, main-v2-compatible publication graphics and auditable exports."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from data_io import map_matrix,SHORT,summary_sem
import visual_grammar as _vg

# CPL asks for 600-dpi figures; the same value sets the resolution of rasterized layers in PDF/SVG.
EXPORT_DPI=600

# Frozen manuscript-wide semantic assignment; redundant line styles preserve
# grayscale legibility where series overlap.
COL={
    'dynamic':'#0072B2','u_output_only':'#E69F00','u_release_only':'#009E73',
    'mean_u_release':'#7A7A7A','static_gain':'#4D4D4D','frozen_profile':'#6F6F6F',
    'baseline':'#2F2F2F','facilitation_only':'#E69F00','depletion_only':'#009E73',
    'u':'#E69F00','r':'#009E73','activity':'#0072B2','co_shift':'#7A7A7A'
}
LS={'dynamic':'-','u_output_only':'--','u_release_only':'-.','mean_u_release':':',
    'static_gain':':','frozen_profile':'--','baseline':'-.'}
REFERENCE='#7A7A7A'


def configure():
    """Apply the frozen final-size manuscript typography and export settings."""
    plt.rcParams.update({
      'font.family':'DejaVu Sans','font.size':6.8,'axes.titlesize':7.2,
      'axes.labelsize':6.8,'xtick.labelsize':6.1,'ytick.labelsize':6.1,'legend.fontsize':6.0,
      'legend.frameon':False,'axes.linewidth':.55,'lines.linewidth':.95,'lines.markersize':3.2,
      'xtick.major.size':2.2,'ytick.major.size':2.2,'xtick.major.width':.55,'ytick.major.width':.55,
      'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
      'svg.fonttype':'none','svg.hashsalt':'gc-stp-final','savefig.dpi':EXPORT_DPI,'figure.dpi':120,
      'axes.titlelocation':'left','mathtext.fontset':'dejavusans',
      'figure.facecolor':'white','savefig.facecolor':'white'})


def figure(number=None,title=None,height=180,footer=None):
    """Create a 180-mm-wide figure using the frozen manuscript defaults.

    ``number``, ``title`` and ``footer`` remain accepted for compatibility with
    older scripts, but presentation-style suptitles/footers are intentionally
    omitted; full titles and provenance belong in captions or local row notes.
    """
    configure()
    fig=plt.figure(figsize=(180/25.4,height/25.4),layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.04,h_pad=.045,wspace=.055,hspace=.06)
    return fig



# Panel-letter case for the target journal: 'lower' (Nature style, matches the manuscript v3)
# or 'upper' (Cell Press style). Change here only; every figure uses panel_letter().
PANEL_LETTER_CASE = 'lower'


def panel_letter(letter):
    """Return the panel label required by the target journal, e.g. "(a)" for CPL."""
    letter = letter.lower() if PANEL_LETTER_CASE == 'lower' else letter.upper()
    return _vg.panel_label(letter)

def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=4,fontsize=7.2)
    ax.annotate(panel_letter(letter), xy=(0, 1), xycoords='axes fraction', xytext=(-4, 3), textcoords='offset points', annotation_clip=False,ha='right',va='bottom',
            fontsize=8.4,fontweight='bold',clip_on=False)
    ax.tick_params(direction='out',labelsize=6.1,length=2.2,width=.55)


def paired(ax,pv,labels=None,colors=None,zero=True):
    if pv.isna().any().any():raise ValueError('Paired plot cannot silently omit missing pairs')
    a=pv.to_numpy(float);x=np.arange(a.shape[1]);j=np.linspace(-.06,.06,len(a))
    ax.add_collection(LineCollection([np.column_stack([x+j[i],row]) for i,row in enumerate(a)],
                                   colors='#9A9A9A',alpha=.30,linewidths=.55,zorder=1))
    for k,c in enumerate(pv.columns):
        color=(colors[k] if colors is not None else COL.get(c,'C0'))
        ax.scatter(x[k]+j,a[:,k],s=10,alpha=.75,color=color,zorder=3)
        ax.plot([x[k]-.13,x[k]+.13],[a[:,k].mean()]*2,color=color,lw=1.55,zorder=4)
    ax.set_xticks(x,labels or [SHORT.get(c,c) for c in pv.columns]);ax.set_xlim(-.35,len(x)-.65)
    if zero:ax.axhline(0,color=REFERENCE,ls=':',lw=.55,alpha=.65,zorder=0)
    ax.autoscale_view(scalex=False)


def forest(ax,labels,means,low,high,primary=None,colors=None):
    n=len(labels);yy=np.arange(n)[::-1]
    primary=[False]*n if primary is None else list(primary)
    for i,(y,m,l,h) in enumerate(zip(yy,means,low,high)):
        if not np.all(np.isfinite([m,l,h])):raise ValueError('Missing forest estimate')
        col=colors[i] if colors else (COL['dynamic'] if primary[i] else '#555555')
        ax.hlines(y,l,h,lw=1.2 if primary[i] else .9,color=col)
        ax.plot([l,h],[y,y],ls='none',marker='|',ms=4,color=col)
        ax.scatter(m,y,s=22 if primary[i] else 15,marker='D' if primary[i] else 'o',color=col,zorder=3)
    ax.axvline(0,color=REFERENCE,ls=':',lw=.6);ax.set_yticks(yy,labels);ax.set_ylim(-.6,n-.4)
    ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    return yy


def line_sem(ax,df,x,value,condition,label=None,scale=1.,individual=False):
    d=df[df.condition==condition].copy()
    s=summary_sem(d,x,value);xx=s[x].to_numpy(float);m=s['mean'].to_numpy(float)*scale;e=s['sem'].to_numpy(float)*scale
    if individual:
        for _,v in d.groupby('seed'):
            v=v.sort_values(x);ax.plot(v[x],v[value]*scale,color=COL[condition],alpha=.13,lw=.55)
    ax.errorbar(xx,m,yerr=e,color=COL[condition],ls=LS.get(condition,'-'),marker='o',ms=3,capsize=2,
                label=label or SHORT.get(condition,condition),lw=1.1)
    return s


def mapplot(ax,df,value,norm,letter,title,cmap='cividis',x='tau_recovery_ms',y='tau_facilitation_ms',
            xs=None,ys=None,anchor=True,annotate=False,hide_y=False):
    mat=map_matrix(df,value,x,y,xs,ys);cm=plt.get_cmap(cmap).copy();cm.set_bad('0.90')
    im=ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)),origin='lower',aspect='auto',
                 norm=norm,cmap=cm,interpolation='none',rasterized=True)
    panel(ax,letter,title)
    ax.set_xticks(range(len(mat.columns)),[f'{v:g}' for v in mat.columns],rotation=45,ha='right')
    ax.set_yticks(range(len(mat.index)),[f'{v:g}' for v in mat.index])
    ax.set_xlabel(r'$\tau_\mathrm{r}$ (ms)')
    if hide_y:ax.tick_params(labelleft=False)
    else:ax.set_ylabel(r'$\tau_\mathrm{f}$ (ms)')
    for sp in ax.spines.values():sp.set_visible(False)
    if anchor and 20. in mat.columns and 1000. in mat.index:
        ix=list(mat.columns).index(20.);iy=list(mat.index).index(1000.)
        ax.add_patch(Rectangle((ix-.5,iy-.5),1,1,fill=False,lw=1.45,ls='-',ec='#202020',zorder=6))
        ax.add_patch(Rectangle((ix-.5,iy-.5),1,1,fill=False,lw=.70,ls='--',ec='white',zorder=7))
    if annotate:
        for iy in range(len(mat.index)):
            for ix in range(len(mat.columns)):
                val=mat.iloc[iy,ix]
                if np.isfinite(val):
                    rgba=cm(norm(val));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(ix,iy,f'{val:g}',ha='center',va='center',fontsize=5.7,
                            color='black' if lum>.53 else 'white')
    return im,mat


def norm_for(values,signed=False,start_zero=True):
    v=np.asarray(values,dtype=float).ravel();v=v[np.isfinite(v)]
    if not len(v):raise ValueError('Cannot normalize an all-missing map')
    if signed:
        maximum=max(float(np.max(np.abs(v))),1e-10)
        return TwoSlopeNorm(vmin=-maximum,vcenter=0.,vmax=maximum)
    lo=0. if start_zero else float(v.min());hi=float(v.max())
    if hi<=lo:hi=lo+1e-10
    return Normalize(lo,hi)


def shared_ylim(axes,series,include_zero=True,pad=.07):
    v=np.concatenate([np.asarray(s,float).ravel() for s in series]);v=v[np.isfinite(v)]
    lo=float(v.min());hi=float(v.max())
    if include_zero:lo=min(0,lo);hi=max(0,hi)
    span=max(hi-lo,1e-8);lim=(lo-pad*span,hi+pad*span)
    for ax in axes:ax.set_ylim(lim)
    return lim


class Export:
    def __init__(self,path,ds,formats=('pdf','svg','png')):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.source=self.path/'source_data';self.source.mkdir(exist_ok=True)
        self.ds=ds;self.formats=tuple(formats);self.records=[];self.assets=[]

    def csv(self,name,df):
        p=self.source/f'{name}.csv';df.to_csv(p,index=False,float_format='%.12g');self.assets.append(p.name);return p

    def npz(self,name,**arrays):
        p=self.source/f'{name}.npz';np.savez_compressed(p,**arrays);self.assets.append(p.name);return p

    def save(self,fig,stem,caption):
        fig.canvas.draw()
        # Check visible text against the physical canvas (excluding clipped data text).
        from matplotlib.text import Text
        renderer=fig.canvas.get_renderer();W,H=fig.bbox.width,fig.bbox.height
        outside=[]
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text() or text.get_clip_on():continue
            bb=text.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W+1 or bb.y1 > H+1:
                outside.append(text.get_text())
        if outside:raise RuntimeError(f'Visible text outside figure canvas: {outside}')
        # v9: report (never raise) text that enters another panel's plotting area.
        try:
            from layout_check import check_all
            layout_warnings = check_all(fig)
        except Exception as exc:  # the self-check must never block a build
            layout_warnings = [f'layout check skipped: {exc}']
        for w in layout_warnings:
            print(f'LAYOUT CHECK {stem}: {w}', flush=True)
        # Save exact physical width, not a bbox_inches='tight' recrop.
        for ext in self.formats:
            kw={'dpi':EXPORT_DPI}
            if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
            if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None,'Creator':'GC-STP supplement v2'}
            if ext=='svg':kw['metadata']={'Date':None}
            fig.savefig(self.path/f'{stem}.{ext}',**kw)
        geometries=[]
        for ax in fig.axes:
            if not ax.get_visible():continue
            geometries.append({'title':ax.get_title(loc='left') or ax.get_title(),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
                               'position':list(ax.get_position().bounds),
                               'color_norms':[{'min':float(im.norm.vmin),'max':float(im.norm.vmax),
                                                'cmap':im.cmap.name} for im in list(ax.images)+[c for c in ax.collections if c.get_array() is not None and c.norm.vmin is not None and c.norm.vmax is not None]]})
        self.records.append({'figure':stem,'caption':caption,'width_mm':round(float(fig.get_size_inches()[0]*25.4),1),
           'height_mm':float(fig.get_size_inches()[1]*25.4),'axes':geometries,
           'source_data':list(self.assets),'inputs':list(self.ds.records.values()),
           'layout_warnings':layout_warnings})
        self.assets=[];self.ds.records={};plt.close(fig);self.finish()
        print(f'Saved {stem}',flush=True)
        return {ext:self.path/f'{stem}.{ext}' for ext in self.formats}

    def finish(self):
        pass
