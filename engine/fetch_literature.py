import json,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
OUT=Path(__file__).parent/'artifacts/literature_review'
def get(r):
    try:
        url='https://api.crossref.org/works/'+urllib.parse.quote(r['doi'],safe='/') if r['doi'] else 'https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':r['title'],'rows':1})
        req=urllib.request.Request(url,headers={'User-Agent':'BuildingWindResearch/1.0'})
        data=json.load(urllib.request.urlopen(req,timeout=40))['message']
        m=data if r['doi'] else data['items'][0]
        return dict(r,registered_title=m.get('title',[''])[0],registered_doi=m['DOI'],journal=m.get('container-title',[''])[0],authors=m.get('author',[]),published=m.get('published',{}),url=m.get('URL'),type=m.get('type'),metadata_source=url)
    except Exception as e:return dict(r,error=str(e))
if __name__=='__main__':
    rows=json.loads((OUT/'candidates.json').read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(get,rows))
    (OUT/'metadata.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:print(r['id'],r.get('registered_doi'),r.get('journal'),r.get('registered_title',r.get('error')),flush=True)
