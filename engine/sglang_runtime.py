"""Prepared configuration binding and private actual-replay complete-call admission."""
from dataclasses import dataclass
import time
import math
from sglang_lifetime import CandidateRejected


@dataclass(frozen=True)
class Configuration:
    shape: tuple
    attention: object
    chunks: object
    native_chunks: object
    verifier: object
    port: bool=False

    def ready(self):
        if self.native_chunks is not None:
            return (set(self.native_chunks)=={1,2,3,4} and self.verifier is not None and
                    all(size in value.graphs for size,value in self.native_chunks.items()))
        return self.chunks is not None and all(size in self.chunks.graphs for size in self.chunks.schedule)


def snapshot(e,port=False):
    return Configuration(e.shape,e.fused_cache_attention,e.chunks,e.native_chunks,e.verifier,port)


def bind(e,cfg):
    e.sg_control.drain()
    if e.sg_timing or cfg.shape != e.shape or not cfg.ready():
        raise RuntimeError('invalid prepared attention configuration binding')
    # One logical transaction after drain; this engine never serves concurrent requests.
    e.fused_cache_attention,e.chunks,e.native_chunks,e.verifier=(
        cfg.attention,cfg.chunks,cfg.native_chunks,cfg.verifier)
    e.sg_bound=cfg


class ReplayCounter:
    def __init__(self,e,cfg,graph,counter):
        self.e,self.cfg,self.graph,self.counter=e,cfg,graph,counter
    def replay(self):
        if self.e.sg_bound is not self.cfg or not self.e.sg_timing:
            raise RuntimeError('replay observer outside its bound warmup call')
        self.graph.replay()
        self.counter[0]+=1


def timed_complete(e,cfg,factory,phase):
    c=e.sg_control
    c.live(phase,5.)
    bind(e,cfg)
    observed=[]
    counter=[0]
    if cfg.port:
        chunks=list(cfg.native_chunks.values()) if cfg.native_chunks is not None else [cfg.chunks]
        for chunk in chunks:
            for size,graph in list(chunk.graphs.items()):
                observed.append((chunk,size,graph))
                chunk.graphs[size]=ReplayCounter(e,cfg,graph,counter)
    captured=e.sg_captures
    e.sg_timing=True
    started=time.perf_counter()
    rows=[]
    gen=None
    try:
        gen=factory()
        for row in gen:
            rows.append(tuple(row))
            c.live()
    finally:
        try:
            try:
                if gen is not None:
                    gen.close()
            finally:
                c.drain()
        finally:
            e.sg_timing=False
            # Proxies own no device storage; owners retain graphs even if poisoned.
            for chunk,size,graph in observed:
                chunk.graphs[size]=graph
    elapsed=time.perf_counter()-started
    if not math.isfinite(elapsed) or elapsed<=0:
        raise CandidateRejected('invalid complete-call timing')
    if e.sg_captures!=captured or e.sg_bound is not cfg or cfg.shape!=e.shape:
        raise RuntimeError('timed request changed capture/configuration')
    c.observed(phase,elapsed)
    return tuple(rows),elapsed,counter[0]


def require_stream(rows,batch,output,vocab):
    if len(rows)!=output or any(len(row)!=batch for row in rows):
        raise CandidateRejected('complete stream shape mismatch')
    if any(type(y)is not int or not 0<=y<vocab for row in rows for y in row):
        raise CandidateRejected('complete stream invalid ID')


def whole_call_admission(e,a,b,prompts,output,factory):
    runs=[timed_complete(e,cfg,factory,phase)
          for cfg,phase in ((a,'base'),(b,'candidate'),(b,'candidate'),(a,'base'))]
    for rows,_,_ in runs:
        require_stream(rows,e.batch,output,e.model.config.vocab_size)
    if not all(run[0]==runs[0][0] for run in runs[1:]):
        raise CandidateRejected('complete greedy stream mismatch')
    if not (runs[1][2]>0 and runs[2][2]>0):
        raise CandidateRejected('no actual port native replay')
    if not max(runs[1][1],runs[2][1])<min(runs[0][1],runs[3][1]):
        raise CandidateRejected('no robust complete-call improvement')
