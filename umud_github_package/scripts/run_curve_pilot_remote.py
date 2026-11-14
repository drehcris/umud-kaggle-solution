"""Run the authorized two-arm pilot on its GPU host with a shared wall-clock deadline.

Stops only child compute, never deletes a pod or data. Cloud billing is managed separately.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/umud')
OUT = Path('/workspace/results/curve_v3')
PYTHON = ROOT / '.venv/bin/python'
ENGINES = ROOT / 'artifacts/gpu_experiments/curve_v3_run_20261022'
DEADLINE = dt.datetime.fromisoformat('2026-10-22T20:00:00+00:00').timestamp()
ENV = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT/'src'),
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
           HF_HOME='/workspace/hf-cache', TORCH_HOME='/workspace/torch-cache',
           UV_CACHE_DIR='/workspace/uv-cache')


def state(status, **kwargs):
    rec = dict(status=status, utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               deadline_utc=dt.datetime.fromtimestamp(DEADLINE,dt.timezone.utc).isoformat(), **kwargs)
    path = OUT/'controller_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(rec,indent=2)+'\n')
    temp.replace(path)
    print(json.dumps(rec),flush=True)


def run(stage, command, env=ENV):
    remaining=DEADLINE-time.time()-120
    if remaining<=0:
        raise TimeoutError('Pilot deadline reached; preserving outputs')
    with (OUT/(stage+'.process.log')).open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state('running',stage=stage,pid=proc.pid)
        try:
            code=proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            raise TimeoutError(f'{stage} exceeded the overall budget; child processes stopped')
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {stage}.process.log')
        state('stage_complete',stage=stage,exit_code=code)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    run('tests',[str(PYTHON),'-m','unittest','discover','-s','tests','-v'])
    run('smoke',[str(PYTHON),'scripts/smoke_curve_gpu.py',
                 '--engine',str(ENGINES/'control/train.py'),
                 '--checkpoint','/workspace/inputs/teacher_v1/checkpoints/fold0.pt',
                 '--out',str(OUT/os.environ.get('UMUD_SMOKE_NAME','smoke'))])
    for arm in ('control','curves'):
        env=dict(ENV,UMUD_INPUT_ROOT='/workspace/inputs',UMUD_OUTPUT_DIR=str(OUT/arm),
                 UMUD_TEMP_DIR=f'/workspace/tmp/curve_v3/{arm}')
        run(arm,[str(PYTHON),str(ENGINES/arm/'train.py')],env)
    run('compare',[str(PYTHON),'scripts/compare_curve_pilot.py','--control',str(OUT/'control'),
                   '--curves',str(OUT/'curves'),'--out',str(OUT/'comparison')])
    state('complete',note='Results preserved. Pod has not been stopped or deleted.')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('failed',error=f'{type(exc).__name__}: {exc}',note='Existing results preserved; no further arms launched.')
        raise
