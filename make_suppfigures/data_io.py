"""Strict archived-data access, provenance, and seed-level transforms.

A selected external results root is authoritative: no silent mixture with bundled
summaries. Illustrative main-v2 reruns are a separate, explicitly labelled source.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from settings import RESULTS_ROOT, RAW_ROOT, CHECKPOINT_ROOT, EXAMPLES_ROOT, ALLOW_RERUN_EXAMPLES, HERE
from dynamics import axial_delta, interp_no_extrap

PHASES = {6:'phase6_corrected_full',7:'phase7_frozen_spatial',8:'phase8_mechanism_map',
 9:'phase9_gain_rescued_depletion',10:'phase10_facilitation_routes',11:'phase11_corrected_analysis',
 12:'phase12_attractor',13:'phase13_attractor_confirmation',14:'phase14_attractor_tether',
 15:'phase15_tether_confirmation',0:'p0_msd'}
LABELS = {'dynamic':'Full','u_output_only':'Output-only','u_release_only':'Release-only',
 'mean_u_release':'Mean release','static_gain':'Static','frozen_profile':'Frozen','baseline':'Baseline',
 'facilitation_only':'Facilitation only','depletion_only':'Depletion only'}
SHORT = {'dynamic':'Full','u_output_only':'Output','u_release_only':'Release',
 'mean_u_release':'Mean rel.','static_gain':'Static','frozen_profile':'Frozen','baseline':'Baseline'}
KINDS = {'activity_shift':'Activity','activity_u_shift':'Activity + u',
 'activity_r_shift':'Activity + R','co_shift':'All states','trace_shift':'Traces only'}


def require_columns(df: pd.DataFrame, columns, name='table'):
    missing = set(columns)-set(df.columns)
    if missing: raise ValueError(f'{name}: missing columns {sorted(missing)}')
    if df.empty: raise ValueError(f'{name}: empty input')


def unique(df: pd.DataFrame, columns, name='table'):
    require_columns(df, columns, name)
    if df.duplicated(columns).any():
        raise ValueError(f'{name}: duplicate keys {columns}; aggregate explicitly, not pivot_table silently')


class DataStore:
    def __init__(self, root=RESULTS_ROOT, examples=EXAMPLES_ROOT, raw_root=RAW_ROOT,
                 checkpoint_root=CHECKPOINT_ROOT, allow_regenerated=ALLOW_RERUN_EXAMPLES):
        root=Path(root).expanduser().resolve()
        # Accept handoff root, results/, or canonical/ as well as bundled_data/.
        for candidate in [root,root/'canonical',root/'results/canonical']:
            if (candidate/'phase6_corrected_full').is_dir():root=candidate;break
        self.root=root
        self.examples=Path(examples).expanduser().resolve()
        raw_root=Path(raw_root).expanduser().resolve() if raw_root is not None else self.root
        for candidate in [raw_root,raw_root/'canonical',raw_root/'results',raw_root/'results/canonical']:
            if (candidate/'phase6_corrected_full').is_dir():
                raw_root=candidate;break
        self.raw_root=raw_root
        self.checkpoint_root=Path(checkpoint_root).expanduser().resolve() if checkpoint_root is not None else None
        self.allow_regenerated=bool(allow_regenerated)
        self.records={}

    def record(self, path, identity='archived analysis'):
        path=Path(path).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        roots=[('results',self.root),('raw',self.raw_root),('examples',self.examples),('package',HERE)]
        if self.checkpoint_root is not None:
            roots.append(('checkpoints',self.checkpoint_root))
        for name,root in roots:
            try: key=f'{name}/{path.relative_to(root).as_posix()}'; break
            except ValueError: continue
        else:key=str(path)
        self.records[key]={'path':key,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'bytes':path.stat().st_size,'identity':identity}
        return path

    def path(self, phase, relative):
        phase=PHASES.get(phase,phase);root=self.root/phase
        candidates=[root/relative,root/phase/relative]
        for p in candidates:
            if p.is_file():return p
        raise FileNotFoundError('Missing required archived input. No data will be fabricated. Tried:\n'+
                                '\n'.join(str(p) for p in candidates))

    def csv(self, phase, relative='analysis/combo_summary.csv', columns=()):
        p=self.record(self.path(phase,relative));df=pd.read_csv(p)
        require_columns(df,columns,str(p));return df

    def json(self,phase,relative):
        return json.loads(self.record(self.path(phase,relative)).read_text(encoding='utf-8'))

    def npz(self,path,identity):
        p=self.record(path,identity)
        with np.load(p,allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def _raw_theta_candidates(self, condition, seed):
        name=f'seed_{seed:04d}.npz'
        if condition=='frozen_profile':
            rels=[
                Path('phase7_frozen_spatial/phase7_frozen_spatial/theta/frozen_profile')/name,
                Path('phase7_frozen_spatial/theta/frozen_profile')/name,
            ]
        elif condition in ['dynamic','static_gain','baseline']:
            rels=[Path('phase6_corrected_full/theta')/condition/name]
        else:
            rels=[Path('phase10_facilitation_routes/combos/tf_1000_tr_20/theta')/condition/name]
        return [self.raw_root/r for r in rels]

    def theta(self,condition='dynamic',seed=0):
        p=next((q for q in self._raw_theta_candidates(condition,seed) if q.is_file()),None)
        identity='archived raw simulation'
        if p is None and self.allow_regenerated and condition!='frozen_profile':
            p=self.examples/'theta'/condition/f'seed_{seed:04d}.npz'
            meta_path=p.with_suffix('.json')
            meta=json.loads(self.record(meta_path,'main-v2 rerun provenance').read_text())
            if meta.get('origin')!='regenerated_exemplar_not_archival_raw' or meta.get('condition')!=condition or meta.get('seed')!=seed:
                raise ValueError('Missing or mismatched explicit rerun provenance')
            identity='main-v2 fixed-seed rerun exemplar; NOT archived raw'
        if p is None or not p.is_file():
            tried='\n'.join(str(q) for q in self._raw_theta_candidates(condition,seed))
            raise FileNotFoundError(
                f'Missing archival raw for {condition}, seed {seed}. Tried:\n{tried}\n'
                'Local-raw mode does not substitute rerun examples unless ALLOW_RERUN_EXAMPLES=True.'
            )
        raw=self.npz(p,identity)
        if str(raw.get('condition'))!=condition or int(raw.get('seed',-1))!=seed:
            raise ValueError('Raw condition / seed identity mismatch')
        require_raw(raw)
        raw['_identity']=identity
        raw['_path']=str(p)
        return raw

    def _branch_path(self,kind='activity_shift',seed=0,delta=11.25):
        sign='p' if delta>=0 else 'm'
        mag=f'{abs(delta):g}'.replace('.','p')
        return self.raw_root/'phase14_attractor_tether'/'branches'/'dynamic'/f'dynamic__seed{seed:04d}__{kind}__{sign}{mag}.npz'

    def branch(self,kind='activity_shift',initial=False,seed=0,delta=11.25):
        p=self._branch_path(kind,seed,delta)
        identity='archived raw simulation'
        if not p.is_file() and self.allow_regenerated and seed==0 and np.isclose(delta,11.25):
            p=self.examples/'branches/dynamic'/f'dynamic__seed0000__{kind}__p11p25.npz'
            meta=json.loads(self.record(p.with_suffix('.json'),'main-v2 rerun provenance').read_text())
            if (meta.get('origin')!='regenerated_exemplar_not_archival_raw' or meta.get('condition')!='dynamic'
                    or meta.get('seed')!=0 or meta.get('shift_kind')!=kind or meta.get('delta_deg')!=11.25):
                raise ValueError('Missing or mismatched surgery exemplar provenance')
            identity='main-v2 fixed-seed surgery rerun exemplar; NOT archived raw'
        if not p.is_file():
            raise FileNotFoundError(
                f'Missing archival Phase-14 branch: {self._branch_path(kind,seed,delta)}\n'
                'Local-raw mode does not substitute rerun examples unless ALLOW_RERUN_EXAMPLES=True.'
            )
        if initial:
            return self._branch_initial_from_raw(p,kind,seed,delta,identity)
        raw=self.npz(p,identity)
        if str(raw.get('condition'))!='dynamic' or int(raw.get('seed',-1))!=seed:
            raise ValueError('Raw Phase-14 branch condition / seed mismatch')
        require_raw(raw)
        raw['_identity']=identity
        raw['_path']=str(p)
        return raw

    def _checkpoint_candidates(self, seed=0, condition='dynamic'):
        name=f'seed_{seed:04d}.pkl'
        candidates=[]
        if self.checkpoint_root is not None:
            candidates += [
                self.checkpoint_root/'checkpoints'/condition/name,
                self.checkpoint_root/condition/name,
            ]
        candidates += [
            self.raw_root/'phase11_anchor_pilot'/'checkpoints'/condition/name,
            self.raw_root/'phase14_attractor_tether'/'checkpoints'/condition/name,
            self.raw_root/'checkpoints'/condition/name,
        ]
        out=[];seen=set()
        for q in candidates:
            q=Path(q)
            if q not in seen:
                out.append(q);seen.add(q)
        return out

    @staticmethod
    def _load_checkpoint_stp_pickle(path):
        """Read only u/resources from the owner's trusted RingNetworkState pickle.
        No simulator code is executed.
        """
        import pickle,sys,types
        saved_model=sys.modules.get('model');saved_state=sys.modules.get('model.state')
        created_model=False
        if saved_state is None:
            if saved_model is None:
                model_mod=types.ModuleType('model');model_mod.__path__=[]
                sys.modules['model']=model_mod;created_model=True
            else:model_mod=saved_model
            state_mod=types.ModuleType('model.state')
            RingNetworkState=type('RingNetworkState',(),{})
            RingNetworkState.__module__='model.state'
            state_mod.RingNetworkState=RingNetworkState
            setattr(model_mod,'state',state_mod)
            sys.modules['model.state']=state_mod
        try:
            with Path(path).open('rb') as f:state=pickle.load(f)
            u=np.asarray(state.u,dtype=float).copy()
            r=np.asarray(state.resources,dtype=float).copy()
            step=int(getattr(state,'step',-1))
        finally:
            if saved_state is None:
                sys.modules.pop('model.state',None)
                if saved_model is not None:
                    sys.modules['model']=saved_model
                    if hasattr(saved_model,'state'):
                        try:delattr(saved_model,'state')
                        except Exception:pass
                elif created_model:
                    sys.modules.pop('model',None)
        if u.ndim!=1 or r.shape!=u.shape:
            raise ValueError(f'Invalid checkpoint STP arrays in {path}')
        return u,r,step

    def _branch_initial_from_raw(self,p,kind,seed,delta,identity):
        # If an exact exported state NPZ exists, use it.
        state_path=p.with_name(p.stem+'_initial_state.npz')
        if state_path.is_file():
            out=self.npz(state_path,identity)
            return out

        # Otherwise recover the original checkpoint u/R and apply only the
        # Phase-14 surgery roll. This is state extraction, not simulation.
        cp=next((q for q in self._checkpoint_candidates(seed,'dynamic') if q.is_file()),None)
        if cp is None:
            tried='\n'.join(str(q) for q in self._checkpoint_candidates(seed,'dynamic'))
            raise FileNotFoundError(
                'Exact spatial STP profiles need the original checkpoint state. '
                'No simulation will be run. Tried:\n'+tried
            )
        self.record(cp,'archived checkpoint state')
        u_before,r_before,step=self._load_checkpoint_stp_pickle(cp)
        n=len(u_before)
        e_bins_float=delta*n/360.0
        if abs(e_bins_float-round(e_bins_float))>1e-9:
            raise ValueError(f'{delta}° is not aligned to the E-cell grid')
        e_bins=int(round(e_bins_float))
        u_after=u_before.copy();r_after=r_before.copy()
        if kind in {'activity_u_shift','co_shift'}:u_after=np.roll(u_after,e_bins)
        if kind in {'activity_r_shift','co_shift'}:r_after=np.roll(r_after,e_bins)
        return {'time_ms':np.array(3000.0),'u_before':u_before,'r_before':r_before,
                'u_after':u_after,'r_after':r_after,
                'theta_e_deg':np.arange(n,dtype=float)*360.0/n,
                'checkpoint_step':np.array(step),
                '_identity':'archived checkpoint state','_path':str(cp)}


def require_raw(raw):
    keys=['sample_times_ms','plfp_gaba_e','theta_e_deg','spike_times_e_ms','spike_angles_e_deg']
    for k in keys:
        if k not in raw:raise ValueError(f'Missing raw array: {k}')
    if len(raw['sample_times_ms'])!=len(raw['plfp_gaba_e']):raise ValueError('Mismatched current time axis')
    if np.any(np.diff(raw['sample_times_ms'])<=0):raise ValueError('Non-monotonic raw times')
    for pop in ['e','i']:
        if len(raw[f'spike_times_{pop}_ms'])!=len(raw[f'spike_angles_{pop}_deg']):raise ValueError('Invalid spike coordinates')


def direction_collapse(df, groups, metrics):
    """Average the two opposite perturbation directions within each seed.

    Do not collapse across displacement magnitudes. Completeness is checked before
    averaging; one surviving direction is not relabelled as a paired average.
    """
    df=df.copy();require_columns(df,list(groups)+['seed','delta_deg']+list(metrics))
    df['abs_delta_deg']=df.delta_deg.abs()
    keys=list(dict.fromkeys(list(groups)+['seed','abs_delta_deg']))
    unique(df,keys+['delta_deg'])
    for key,d in df.groupby(keys,dropna=False):
        v=np.sort(d.delta_deg.to_numpy(float))
        if len(v)!=2 or v[0]>=0 or not np.isclose(v[0],-v[1]):
            raise ValueError(f'Incomplete +/- direction pair at {key}')
    out=df.groupby(keys,as_index=False)[list(metrics)].mean()
    out['n_directions']=2
    return out


def collapse_noise(df,metrics,expected_draws=8):
    keys=['condition','seed','noise_pa']
    unique(df,keys+['noise_draw'])
    sizes=df.groupby(keys).noise_draw.nunique()
    if not np.all(sizes==expected_draws):raise ValueError('Incomplete nested noise draws')
    out=df.groupby(keys,as_index=False)[list(metrics)].mean();out['n_noise_draws']=expected_draws
    return out


def paired_matrix(df, group, value, order, expected_n=None):
    unique(df,['seed',group]);p=df.pivot(index='seed',columns=group,values=value).reindex(columns=order)
    if p.isna().any().any():raise ValueError(f'Incomplete paired seed matrix for {value}')
    if expected_n is not None and len(p)!=expected_n:raise ValueError(f'Expected {expected_n} seeds; got {len(p)}')
    return p.sort_index()


def summary_sem(df,x,value):
    # df must already contain one value per independent seed at each x.
    unique(df,['seed',x]);s=df.groupby(x)[value].agg(['mean','sem','count']).reset_index()
    s=s.rename(columns={'count':'n_seeds'});return s


def phase14_direction_collapsed(timeseries,displacement=11.25,step_ms=25.,kind='activity_shift'):
    """Axial, matched-reference displacement; joint time support, never extrapolated."""
    curves=[]
    values=['activity_rel_deg','u_rel_deg','r_rel_deg','activity_minus_u_deg','activity_minus_r_deg']
    for seed in sorted(timeseries.seed.unique()):
        ref=timeseries[(timeseries.seed==seed)&(timeseries.shift_kind=='no_shift')&np.isclose(timeseries.delta_deg,0)]
        unique(ref,['elapsed_ms'],'no-shift trace');directions=[]
        for delta in [displacement,-displacement]:
            br=timeseries[(timeseries.seed==seed)&(timeseries.shift_kind==kind)&np.isclose(timeseries.delta_deg,delta)]
            unique(br,['elapsed_ms'],'shift trace')
            m=br.merge(ref[['elapsed_ms','activity_axis_deg','u_axis_deg','r_axis_deg']],
                       on='elapsed_ms',suffixes=('','_ref'),validate='one_to_one').sort_values('elapsed_ms')
            if m.empty:raise ValueError('No common branch/reference observations')
            sign=np.sign(delta);out={'elapsed_ms':m.elapsed_ms.to_numpy(float)}
            for state in ['activity','u','r']:
                out[f'{state}_rel_deg']=sign*axial_delta(m[f'{state}_axis_deg'],m[f'{state}_axis_deg_ref'])
            for state in ['u','r']:out[f'activity_minus_{state}_deg']=sign*m[f'activity_minus_{state}_deg'].to_numpy(float)
            d=pd.DataFrame(out);d['direction']=int(sign);directions.append(d)
        both=pd.concat(directions)
        # A time point must be observed in both directions, not just one.
        count=both.groupby('elapsed_ms').direction.nunique()
        both=both[both.elapsed_ms.isin(count[count==2].index)]
        s=both.groupby('elapsed_ms',as_index=False)[values].mean();s['seed']=seed;curves.append(s)
    start=max(d.elapsed_ms.min() for d in curves);end=min(d.elapsed_ms.max() for d in curves)
    grid=np.arange(np.ceil(start/step_ms)*step_ms,np.floor(end/step_ms)*step_ms+.1,step_ms)
    if len(grid)<2:raise ValueError('No shared time support')
    rows=[]
    for d in curves:
        r={'seed':int(d.seed.iloc[0]),'elapsed_ms':grid}
        for v in values:r[v]=interp_no_extrap(d.elapsed_ms.to_numpy(),d[v].to_numpy(),grid)
        rows.append(pd.DataFrame(r))
    return pd.concat(rows,ignore_index=True)


def integrated_anchor(seed_df):
    d=seed_df[(seed_df.combo_id=='tf_1000_tr_20')&(seed_df['mode']=='dynamic')].copy()
    require_columns(d,['paired_valid','seed','noise_pa'])
    if not np.all(d.paired_valid==1):raise ValueError('Anchor has invalid pairs; revise complete-case specification explicitly')
    unique(d,['seed','noise_pa']);rows=[]
    for seed,g in d.groupby('seed'):
        g=g.sort_values('noise_pa');x=g.noise_pa.to_numpy(float)
        if not np.array_equal(x,np.array([0,30,60,90])):raise ValueError('Incomplete anchor noise grid')
        bt=trapezoid(g.temporal_noise_abs_delta_deg,x)/(x[-1]-x[0])
        bs=trapezoid(g.static_noise_abs_delta_deg,x)/(x[-1]-x[0])
        rows.append({'seed':seed,'Full':bt,'Static':bs,'static_minus_full_deg':bs-bt})
    return pd.DataFrame(rows)


def field_rows(df):
    summary=df[df.seed.astype(str)=='summary'].copy()
    individual=df[df.seed.astype(str)!='summary'].copy()
    individual['seed']=pd.to_numeric(individual.seed,errors='raise').astype(int)
    unique(individual,['condition','seed']);unique(summary,['condition'])
    return individual,summary


def map_matrix(df,value,x='tau_recovery_ms',y='tau_facilitation_ms',xs=None,ys=None):
    unique(df,[x,y]);xs=sorted(df[x].unique()) if xs is None else list(xs)
    ys=sorted(df[y].unique()) if ys is None else list(ys)
    m=df.pivot(index=y,columns=x,values=value).reindex(index=ys,columns=xs)
    return m
