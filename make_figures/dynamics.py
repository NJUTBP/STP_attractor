"""Numerical transforms with explicit units, axial geometry and no extrapolation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal


def axial_delta(a, b=0.):
    """Difference on the two-bump (180 degree, not 360 degree) manifold."""
    return (np.asarray(a)-np.asarray(b)+90.)%180.-90.


def axial_unwrap(x):
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(x)*2)))/2


def interp_no_extrap(x, y, grid):
    x,y,grid=map(np.asarray,(x,y,grid))
    if len(x)!=len(y) or len(x)<2 or not np.all(np.diff(x)>0):
        raise ValueError('Interpolation requires unique increasing sample times.')
    if grid.min()<x.min()-1e-8 or grid.max()>x.max()+1e-8:
        raise ValueError('Refusing endpoint extrapolation; shorten the common time grid.')
    return np.interp(grid,x,y)


def welch_proxy(raw, start_ms=1000.,end_ms=8000.):
    t=np.asarray(raw['sample_times_ms']);x=np.asarray(raw['plfp_gaba_e'])
    step=np.diff(t)
    if not np.allclose(step,5.,atol=1e-8):raise ValueError('Canonical PSD requires 200 Hz sampling.')
    mask=(t>=start_ms)&(t<=end_ms)
    if mask.sum()<256 or t[-1]<end_ms:raise ValueError('Insufficient PSD window; do not silently shorten it.')
    f,p=signal.welch(x[mask],fs=200.,window=signal.get_window('hamming',256),
                     nperseg=256,noverlap=128,nfft=1024)
    m=(f>0)&(f<=50)
    return f,p,float(f[m][np.argmax(p[m])])


def theta_band(raw):
    """Offline visualization only; zero-phase filter is not a causal timing measure."""
    x=np.asarray(raw['plfp_gaba_e']); t=np.asarray(raw['sample_times_ms'])
    if not np.allclose(np.diff(t),5.,atol=1e-8):raise ValueError('Expected 200 Hz')
    sos=signal.butter(3,[4.,12.],btype='bandpass',fs=200.,output='sos')
    return signal.sosfiltfilt(sos,x)


def cycle_waveforms(raw, points=101):
    """Raw-current waveforms aligned to rising zero crossings of 4–12 Hz proxy.
    Use crossings within 2–7 s (one-second filter-edge guard after 1 s exclusion).
    Each cycle is demeaned, not amplitude-normalized; phase is descriptive only.
    """
    t=np.asarray(raw['sample_times_ms']);x=np.asarray(raw['plfp_gaba_e'])
    b=theta_band(raw)
    inds=np.flatnonzero((b[:-1]<0)&(b[1:]>=0))
    crossings=t[inds] + (t[inds+1]-t[inds])*(-b[inds])/(b[inds+1]-b[inds])
    phase=np.linspace(0,1,points); out=[]; bounds=[]
    for a,z in zip(crossings[:-1],crossings[1:]):
        if a<2000 or z>7000 or not (1000/12 <= z-a <= 1000/4):continue
        row=np.interp(a+phase*(z-a),t,x)
        out.append(row-row.mean());bounds.append([a,z])
    if len(out)<5:raise ValueError('Too few admissible cycles; no synthesized phase average.')
    return phase,np.asarray(out),np.asarray(bounds)


def spatial_rates(raw, time_edges_ms, angle_edges_deg, folded=False, population='e'):
    """Hz per neuron per bin. Denominator includes silent neurons."""
    time_edges_ms=np.asarray(time_edges_ms);angle_edges_deg=np.asarray(angle_edges_deg)
    angles=np.asarray(raw[f'spike_angles_{population}_deg'])
    if population=='e': positions=np.asarray(raw['theta_e_deg'])
    else: positions=np.arange(256)*360/256
    if folded:angles=angles%180;positions=positions%180
    counts,_,_=np.histogram2d(angles,raw[f'spike_times_{population}_ms'],
                            bins=[angle_edges_deg,time_edges_ms])
    neurons=np.histogram(positions,bins=angle_edges_deg)[0]
    denominator=neurons[:,None]*np.diff(time_edges_ms)[None,:]/1000
    rates=np.divide(counts,denominator,out=np.full_like(counts,np.nan),where=denominator>0)
    return rates,counts,neurons


def pav(raw,times_ms,window_ms=400.):
    """Trailing, spike-count weighted second circular moment of E spikes."""
    st=np.asarray(raw['spike_times_e_ms']);ang=np.asarray(raw['spike_angles_e_deg'])
    order=np.argsort(st,kind='stable');st=st[order];ang=ang[order]
    z=np.exp(2j*np.deg2rad(ang));cs=np.r_[0j,np.cumsum(z)]
    times_ms=np.asarray(times_ms)
    lo=np.searchsorted(st,times_ms-window_ms,side='left')
    hi=np.searchsorted(st,times_ms,side='left')
    sums=cs[hi]-cs[lo];out=(np.rad2deg(np.angle(sums))/2)%180
    out[hi==lo]=np.nan
    return out


def replay_stp(raw, sample_ms, u0=None,r0=None,start_ms=0.,U=.4,tau_f=1000.,tau_r=20.,dt_ms=.02,mode=None):
    """Exact event replay of the CURRENT model, not a fitted profile.

    The simulator recovers a full dt before spike-triggered updates; timestamps
    denote step starts. The effective update time is therefore spike_time+dt.
    A supplied checkpoint is the post-intervention state at start_ms.
    Returns full arrays [sample, neuron]. Validate against stored axial traces.
    """
    times=np.asarray(sample_ms,dtype=float);n=len(raw['theta_e_deg'])
    if len(times)==0 or times.min()<start_ms or np.any(np.diff(times)<=0):raise ValueError('Invalid replay times')
    if mode is None: mode=str(raw['condition'])
    if mode not in ['dynamic','u_output_only','u_release_only']:
        raise ValueError(f'STP replay is not defined here for {mode}')
    u0=np.full(n,U) if u0 is None else np.asarray(u0,dtype=float)
    r0=np.ones(n) if r0 is None else np.asarray(r0,dtype=float)
    if u0.shape!=(n,) or r0.shape!=(n,):raise ValueError('Bad initial state size')
    ids=np.rint(np.asarray(raw['spike_angles_e_deg'])*n/360).astype(int)%n
    st=np.asarray(raw['spike_times_e_ms'])+dt_ms
    keep=(st>start_ms+1e-9)&(st<=times[-1]+1e-8)
    ids,st=ids[keep],st[keep]
    u=np.empty((len(times),n));r=np.empty_like(u)
    for i in range(n):
        event=np.sort(st[ids==i])
        knots=np.r_[start_ms,event]
        us=np.empty(len(knots));rs=np.empty(len(knots));us[0]=u0[i];rs[0]=r0[i]
        for j in range(1,len(knots)):
            d=knots[j]-knots[j-1]
            before=U+(us[j-1]-U)*np.exp(-d/tau_f)
            us[j]=before+U*(1-before)
            depleted=us[j] if mode in ['dynamic','u_release_only'] else U
            rs[j]=(1+(rs[j-1]-1)*np.exp(-d/tau_r))*(1-depleted)
        k=np.searchsorted(knots,times+1e-8,side='right')-1
        d=np.maximum(0,times-knots[k])
        u[:,i]=U+(us[k]-U)*np.exp(-d/tau_f)
        r[:,i]=1+(rs[k]-1)*np.exp(-d/tau_r)
    return u,r


def state_axis(weights,positions):
    weights=np.asarray(weights)
    z=weights@np.exp(2j*np.deg2rad(positions))
    out=np.rad2deg(np.angle(z))/2%180
    out[np.sum(weights,axis=1)<=1e-12]=np.nan
    return out


def collapse_discovery(table,delta=11.25,grid=None):
    """Paired no-shift subtraction, unwrap, interpolate INSIDE support,
    sign-align, average +/- branches within seed before statistics.
    Values are displacements relative to same-seed no-shift, in degrees.
    """
    if grid is None:grid=np.arange(25.,1975.+1,25.)
    rows=[]
    fields={'activity_axis_deg':'activity','u_axis_deg':'u','r_axis_deg':'resource'}
    for seed,g in table.groupby('seed'):
        baseline=g[(g.shift_kind=='no_shift')&(g.delta_deg==0)].sort_values('elapsed_ms')
        if baseline.empty:raise ValueError(f'No no-shift reference for seed {seed}')
        t0=baseline.elapsed_ms.to_numpy()
        ref={k:interp_no_extrap(t0,axial_unwrap(baseline[k]),grid) for k in fields}
        for kind,b in g[np.isclose(g.delta_deg.abs(),delta)].groupby('shift_kind'):
            signs=[]
            for sign,sg in b.groupby(np.sign(b.delta_deg)):
                sg=sg.sort_values('elapsed_ms')
                d={k:sign*axial_delta(interp_no_extrap(sg.elapsed_ms,axial_unwrap(sg[k]),grid),ref[k]) for k in fields}
                signs.append(d)
            if len(signs)!=2:raise ValueError(f'Missing paired shift directions: {seed}, {kind}')
            for j,t in enumerate(grid):
                row={'seed':seed,'shift_kind':kind,'elapsed_ms':t,'abs_delta_deg':delta}
                row.update({v:np.mean([s[k][j] for s in signs]) for k,v in fields.items()})
                rows.append(row)
    out=pd.DataFrame(rows)
    out['mismatch']=out.activity-out.u
    # A 400-ms trailing PAV is mixed pre/post for times <400 ms.
    out.loc[out.elapsed_ms<400,['activity','mismatch']]=np.nan
    return out


def seed_mean_band(table,value,group='elapsed_ms',n_boot=4000,seed=173):
    """Pointwise 95% percentile bootstrap, resampling independent seeds only."""
    p=table.pivot(index='seed',columns=group,values=value).sort_index(axis=1)
    arr=p.to_numpy();valid=np.isfinite(arr).all(axis=0)
    mean=np.full(arr.shape[1],np.nan);low=mean.copy();high=mean.copy()
    if valid.any():
        rng=np.random.default_rng(seed)
        draws=rng.integers(0,len(arr),size=(n_boot,len(arr)))
        boot=arr[:,valid][draws].mean(axis=1)
        mean[valid]=arr[:,valid].mean(axis=0)
        low[valid],high[valid]=np.quantile(boot,[.025,.975],axis=0)
    return p.columns.to_numpy(dtype=float),mean,low,high,len(arr)


def paired_matrix(table,groups,value='R_2000ms',column='condition'):
    subset=table[table[column].isin(groups)]
    if subset.duplicated(['seed',column]).any():raise ValueError('Duplicate seed/group rows; collapse directions and deltas first.')
    pivot=subset.pivot(index='seed',columns=column,values=value).reindex(columns=groups)
    if pivot.isna().any().any():raise ValueError('Incomplete seed pairs; refusing silent unpairing.')
    return pivot
