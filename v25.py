"""V2.5 unified entry: six configurations, one research workflow."""
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def main():
    args=sys.argv[1:]
    if not args or args[0] in ('--help','-h'):
        print('V2.5: status | predict --method mlp|unet_si|unet|fno|ufno|mlp_data --dims W L H --z Z --output DIR | baseline prepare|run|report|finalize | adapt prepare|run|report|status')
        print('Optional global --results DIR. Predict/status default to registered reference results; baseline/adapt default to new V2.5 artifacts.')
        return
    results=None
    if '--results' in args:
        i=args.index('--results');results=args[i+1];del args[i:i+2]
    results=results or CONFIG['reference_results' if args[0] in ('predict','status') else 'new_results']
    if args[0] in ('baseline','adapt') and Path(results).resolve()==Path(CONFIG['reference_results']).resolve():
        raise SystemExit('Historical reference results are read-only. Select a new V2.5 results directory.')
    os.environ['WINDFIELD_V25_RESULTS']=str(Path(results).resolve())
    if '--method' in args:
        i=args.index('--method')+1
        if args[i] not in CONFIG['models']: raise SystemExit('Choose a V2.5 configuration: '+', '.join(CONFIG['models']))
        args[i]=CONFIG['models'][args[i]]
    sys.path.insert(0,str(ROOT/'engine'))
    import research
    sys.argv=[str(ROOT/'v25.py'),*args]
    research.main()
    if args[0]=='predict':
        out=Path(args[args.index('--output')+1]);p=out/'provenance.json'
        data=json.loads(p.read_text(encoding='utf-8'));data['project_version']='2.5'
        data['configuration']=next(k for k,v in CONFIG['models'].items() if v==data['method'])
        p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
