"""Run unchanged official framework and summarize timings. Run in competition env."""
import argparse,json,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/participant/pingpang'));p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve(); inp=(a.input or root/'public_data').resolve(); out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(root/'integrity.py'),'--verify'],cwd=root,check=True)
    command=[sys.executable,str(root/'run.py'),'--input',str(inp),'--output',str(out),'--solution','participant.solution:Solution','--decoder','gpu']
    r=subprocess.run(command,cwd=root)
    status_path=out/'run_status.json'
    if not status_path.exists():raise RuntimeError(f'No run_status.json; exit={r.returncode}')
    s=json.loads(status_path.read_text(encoding='utf-8'));videos=s.get('videos',[])
    sums={k:sum(v.get(k,0) for v in videos) for k in ('decode_sec','solution_sec','write_sec')}
    report={'exit_code':r.returncode,'status':s.get('status'),'timing':s.get('timing'),'performance':s.get('performance'),'stage_sums_seconds':sums,'videos':len(videos),'failed_or_skipped':[v.get('video_id') for v in videos if v.get('status')!='ok'],'note':'Run once in a fresh output folder for cold build, then reuse the same folder for warm cache. Archive both summaries separately.'}
    (out/'e_profile_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
    if r.returncode or report['failed_or_skipped'] or not videos:raise SystemExit(1)
if __name__=='__main__':main()
