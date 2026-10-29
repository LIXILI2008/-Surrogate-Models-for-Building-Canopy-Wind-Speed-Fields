"""Finish reports/figures after the already-running training process exits."""
import argparse
import os
import time
import traceback
from study import read, write
from adaptation import OUT, report
from bench.runner import process_alive


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-pid',type=int,required=True);args=ap.parse_args()
    record=OUT/'postprocess_status.json'
    if record.exists():
        old=read(record)
        if old.get('pid')!=os.getpid() and process_alive(old.get('pid',-1)):
            raise RuntimeError('Postprocessing companion already running')
    write(record,dict(state='WAITING_FOR_TRAINING',pid=os.getpid(),training_pid=args.training_pid))
    try:
        while process_alive(args.training_pid):time.sleep(5)
        state=read(OUT/'status.json')
        if state.get('state')!='EXPERIMENTS_COMPLETE':
            raise RuntimeError(f'Training did not finish successfully: {state}')
        write(record,dict(state='GENERATING_REPORTS',pid=os.getpid()))
        report()
        from adaptation_figures import main as figures
        figures()
        write(record,dict(state='COMPLETE',pid=os.getpid(),finished=time.time(),report=str(OUT/'REPORT.md'),gallery=str(OUT/'diagnostics/index.html')))
    except BaseException:
        write(record,dict(state='FAILED',pid=os.getpid(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
