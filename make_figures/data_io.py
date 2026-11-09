"""Validated, provenance-aware inputs. Never synthesize missing scientific data."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PATHS = {
 'theta6': 'phase6_corrected_full/analysis/theta_summary.csv',
 'theta7': 'phase7_frozen_spatial/analysis/theta_per_seed.csv',
 'map8': 'phase8_mechanism_map/analysis/combo_summary.csv',
 'routes': 'phase10_facilitation_routes/analysis/combo_summary.csv',
 'routes_seed': 'phase10_facilitation_routes/analysis/per_seed_summary.csv',
 'traces14': 'phase14_attractor_tether/analysis/phase14_trace_mismatch_timeseries.csv',
 'seed14': 'phase14_attractor_tether/analysis/phase14_seed_metrics.csv',
 'seed15': 'phase15_tether_confirmation/analysis/phase15_seed_metrics.csv',
 'contrasts15': 'phase15_tether_confirmation/analysis/phase15_confirmatory_contrasts.csv',
 'retention15': 'phase15_tether_confirmation/analysis/phase15_trace_retention_seed.csv',
 'retention_summary15': 'phase15_tether_confirmation/analysis/phase15_trace_retention_summary.csv',
 'noise13': 'phase13_attractor_confirmation/analysis/phase13_summary.json',
 'msd': 'p0_msd/phase13/phase13_msd_curves.csv',
}
LABELS = {'dynamic':'Full', 'static_gain':'Static', 'baseline':'Baseline',
          'frozen_profile':'Frozen', 'u_output_only':'Output-only',
          'u_release_only':'Release-only', 'mean_u_release':'Mean release'}
SURGERY = {'activity_shift':'Activity', 'activity_u_shift':'Activity + u',
           'activity_r_shift':'Activity + R', 'co_shift':'Co-shift'}

@dataclass
class RawTrial:
    arrays: dict
    path: Path
    origin: str
    metadata: dict
    def __getitem__(self,key): return self.arrays[key]

class DataStore:
    def __init__(self, results_root: Path, examples: Path, raw_root: Path | None=None,
                 checkpoint_root: Path | None=None, seed: int=0, allow_regenerated: bool=False):
        self.root=Path(results_root).resolve()
        self.examples=Path(examples).resolve()
        self.raw_root=Path(raw_root).resolve() if raw_root else self.root
        self.checkpoint_root=Path(checkpoint_root).resolve() if checkpoint_root else None
        self.seed=seed
        self.allow_regenerated=allow_regenerated
        self.used={}
        self.cache={}
        if not self.root.is_dir(): raise FileNotFoundError(self.root)

    def record(self,path:Path,origin:str,**metadata):
        # Relative paths make the exported manifest portable.
        roots=[('results',self.root),('raw',self.raw_root),('examples',self.examples),
               ('package',Path(__file__).resolve().parent)]
        if self.checkpoint_root is not None:
            roots.append(('checkpoints', self.checkpoint_root))
        label=path.name
        for prefix,root in roots:
            try: label=prefix+'/'+str(path.relative_to(root)); break
            except ValueError: pass
        self.used[label]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'origin':origin,**metadata}

    def table(self,key:str):
        if key in self.cache:
            return self.cache[key].copy()

        rel = Path(PATHS[key])

        # Primary convention = owner's real local results tree:
        #   results/phase7_frozen_spatial/analysis/...
        #   results/p0_msd/phase13/...
        #
        # The handoff bundle introduced two packaging-only duplicated folders:
        #   phase7_frozen_spatial/phase7_frozen_spatial/...
        #   p0_msd/p0_msd/...
        #
        # Try the local path first, then only these known handoff variants.
        candidates = [self.root / rel]
        if rel.parts and rel.parts[0] == 'phase7_frozen_spatial':
            candidates.append(
                self.root / 'phase7_frozen_spatial' / rel
            )
        if rel.parts and rel.parts[0] == 'p0_msd':
            candidates.append(
                self.root / 'p0_msd' / rel
            )

        # De-duplicate while preserving priority.
        unique = []
        seen = set()
        for q in candidates:
            q = Path(q)
            if q not in seen:
                unique.append(q)
                seen.add(q)
        candidates = unique

        p = next((q for q in candidates if q.is_file()), None)
        if p is None:
            raise FileNotFoundError(
                'Missing required archival table. Tried:\n' +
                '\n'.join(str(q) for q in candidates)
            )

        self.record(p, 'archived_summary')
        value = json.loads(p.read_text(encoding='utf-8')) if p.suffix == '.json' else pd.read_csv(p)
        self.cache[key] = value
        return value.copy()

    def _read_raw(self,path:Path,origin:str,condition:str):
        with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
        required=['sample_times_ms','plfp_gaba_e','spike_times_e_ms','spike_angles_e_deg',
                  'spike_times_i_ms','spike_angles_i_deg','theta_e_deg','condition','seed']
        absent=[k for k in required if k not in a]
        if absent: raise ValueError(f'{path}: missing {absent}')
        if str(a['condition'])!=condition or int(a['seed'])!=self.seed:
            raise ValueError(f'Condition/seed mismatch in {path}')
        t=a['sample_times_ms']
        if not(len(t)>2 and np.all(np.diff(t)>0) and np.isfinite(a['plfp_gaba_e']).all()):
            raise ValueError(f'Invalid raw sampling in {path}')
        if 'noise_pa' in a and float(a['noise_pa'])!=0: raise ValueError('Theta/surgery exemplar must be zero noise')
        meta={}
        sidecar=path.with_suffix('.json')
        if sidecar.exists():
            meta=json.loads(sidecar.read_text(encoding='utf-8'))
            self.record(sidecar,origin)
        if origin=='regenerated_exemplar' and meta.get('origin')!='regenerated_exemplar_not_archival_raw':
            raise ValueError(f'Unlabelled regenerated raw: {path}')
        self.record(path,origin,condition=condition,seed=self.seed)
        return RawTrial(a,path,origin,meta)

    def theta(self,condition:str):
        k=('raw_theta',condition)
        if k in self.cache:return self.cache[k]
        s=f'seed_{self.seed:04d}.npz'
        if condition=='frozen_profile':
            rel=Path('phase7_frozen_spatial/theta/frozen_profile')/s
        elif condition in ['dynamic','static_gain','baseline']:
            rel=Path('phase6_corrected_full/theta')/condition/s
        else:
            rel=Path('phase10_facilitation_routes/combos/tf_1000_tr_20/theta')/condition/s
        # Support the common flattened Phase-7 directory but never arbitrary recursive matches.
        candidates=[self.raw_root/rel]
        if condition=='frozen_profile':
            candidates += [
                self.raw_root/'phase7_frozen_spatial/phase7_frozen_spatial/theta/frozen_profile'/s
            ]
        p=next((p for p in candidates if p.is_file()),None)
        origin='archived_raw'
        if p is None and self.allow_regenerated:
            p=self.examples/'theta'/condition/s;origin='regenerated_exemplar'
        if p is None or not p.is_file():
            raise FileNotFoundError(f'No real raw for {condition}. Expected {candidates}; regenerated exemplars not available/allowed.')
        raw=self._read_raw(p,origin,condition);self.cache[k]=raw
        return raw

    BRANCH_FOLDERS={'phase14':'phase14_attractor_tether','phase15':'phase15_tether_confirmation'}

    def branch(self,kind:str,experiment:str='phase14',sign:str='p',magnitude:str='11p25'):
        """One archived continuation. experiment='phase14' is the initial set (seeds 0-7; the archived Fig. 3
        exemplar is seed 0), 'phase15' the confirmation set (seeds 200-215) summarized in Fig. 3(d)."""
        k=('branch',kind,experiment,sign,magnitude)
        if k in self.cache:return self.cache[k]
        name=(f'dynamic__seed{self.seed:04d}__no_shift.npz' if kind=='no_shift'
              else f'dynamic__seed{self.seed:04d}__{kind}__{sign}{magnitude}.npz')
        p=self.raw_root/self.BRANCH_FOLDERS[experiment]/'branches/dynamic'/name
        origin='archived_raw'
        if not p.exists() and self.allow_regenerated:
            p=self.examples/'branches/dynamic'/name;origin='regenerated_exemplar'
        if not p.exists():raise FileNotFoundError(f'Missing raw surgery exemplar {p}')
        raw=self._read_raw(p,origin,'dynamic');self.cache[k]=raw
        return raw

    def _checkpoint_candidates(self, condition='dynamic'):
        """Fixed, non-recursive locations used by the original Phase-11/14 pipeline."""
        seed_name=f"seed_{self.seed:04d}.pkl"
        candidates=[]
        if self.checkpoint_root is not None:
            candidates.append(self.checkpoint_root/'checkpoints'/condition/seed_name)
            candidates.append(self.checkpoint_root/condition/seed_name)
        candidates += [
            self.raw_root/'phase11_anchor_pilot'/'checkpoints'/condition/seed_name,
            self.raw_root/'phase14_attractor_tether'/'checkpoints'/condition/seed_name,
            self.raw_root/'checkpoints'/condition/seed_name,
        ]
        # de-duplicate while keeping order
        out=[]
        seen=set()
        for q in candidates:
            q=Path(q)
            if q not in seen:
                out.append(q);seen.add(q)
        return out

    @staticmethod
    def _load_checkpoint_stp_pickle(path:Path):
        """Read only u/resources from the owner's trusted RingNetworkState pickle.

        The compatibility shim avoids importing/running the simulator. It exists
        only so plotting can use the original saved checkpoint state.
        """
        import pickle, sys, types
        saved_model=sys.modules.get('model')
        saved_state=sys.modules.get('model.state')
        created_model=False
        if saved_state is None:
            if saved_model is None:
                model_mod=types.ModuleType('model'); model_mod.__path__=[]
                sys.modules['model']=model_mod; created_model=True
            else:
                model_mod=saved_model
            state_mod=types.ModuleType('model.state')
            RingNetworkState=type('RingNetworkState', (), {})
            RingNetworkState.__module__='model.state'
            state_mod.RingNetworkState=RingNetworkState
            setattr(model_mod,'state',state_mod)
            sys.modules['model.state']=state_mod
        try:
            with Path(path).open('rb') as f:
                state=pickle.load(f)
            u=np.asarray(state.u,dtype=float).copy()
            r=np.asarray(state.resources,dtype=float).copy()
            step=int(getattr(state,'step',-1))
        finally:
            if saved_state is None:
                sys.modules.pop('model.state',None)
                if saved_model is not None:
                    sys.modules['model']=saved_model
                    if hasattr(saved_model,'state'):
                        try: delattr(saved_model,'state')
                        except Exception: pass
                elif created_model:
                    sys.modules.pop('model',None)
        if u.ndim!=1 or r.shape!=u.shape:
            raise ValueError(f'Invalid checkpoint STP arrays in {path}')
        return u,r,step

    def branch_initial(self,kind:str):
        """Return exact post-surgery u/R state without rerunning the network."""
        raw=self.branch(kind)

        # First support an explicitly saved NPZ checkpoint state.
        p=raw.path.with_name(raw.path.stem+'_initial_state.npz')
        if p.is_file():
            self.record(p,raw.origin)
            with np.load(p,allow_pickle=False) as z:
                return {k:z[k] for k in z.files}

        # Original project raw normally stores the prehistory checkpoint as pickle.
        cp=next((q for q in self._checkpoint_candidates('dynamic') if q.is_file()),None)
        if cp is None:
            tried='\n'.join(str(q) for q in self._checkpoint_candidates('dynamic'))
            raise FileNotFoundError(
                'Exact spatial STP replay needs either *_initial_state.npz or the '
                'original Phase-11 checkpoint pickle. Tried:\n'+tried
            )
        self.record(cp,'archived checkpoint state')
        u_before,r_before,step=self._load_checkpoint_stp_pickle(cp)

        # Apply exactly the Phase-14 spatial surgery to the STP variables.
        n=len(u_before)
        delta_deg=11.25
        e_bins_float=delta_deg*n/360.0
        if abs(e_bins_float-round(e_bins_float))>1e-9:
            raise ValueError('11.25° is not aligned to the E-cell grid')
        e_bins=int(round(e_bins_float))
        u_after=u_before.copy(); r_after=r_before.copy()
        if kind in {'activity_u_shift','co_shift'}:
            u_after=np.roll(u_after,e_bins)
        if kind in {'activity_r_shift','co_shift'}:
            r_after=np.roll(r_after,e_bins)

        return {
            'time_ms': np.array(3000.0),
            'u_before':u_before,
            'r_before':r_before,
            'u_after':u_after,
            'r_after':r_after,
            'theta_e_deg':np.arange(n,dtype=float)*360.0/n,
            'checkpoint_step':np.array(step),
        }
