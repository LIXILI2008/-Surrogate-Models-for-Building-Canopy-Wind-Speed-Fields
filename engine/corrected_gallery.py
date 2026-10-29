"""Regenerate completed-method gallery at equal physical aspect ratio."""
from study import ART, METHODS, SEEDS, read
from reporting import galleries

p=read(ART/'protocol.json')
methods=[m for m in METHODS if all((ART/f'production/{m}/seed_{s}/evaluation.json').exists() for s in SEEDS)]
out=ART/'interim_completed_models'/'gallery_physical_scale'
galleries(p,methods=methods,out=out,physical_scale=True)
# Keep prior PNGs unchanged and point its index to the physically scaled replacement.
old=ART/'interim_completed_models'/'gallery'/'index.html'
if old.exists():
    s=old.read_text(encoding='utf-8')
    note='<p><strong>比例提示：</strong>原预览图将 x/L 与 y/W 分别归一化后铺满面板，平面几何会变形。请查看按米等比例显示的修正版：<a href="../gallery_physical_scale/index.html">物理比例修正版图集</a>。原PNG保留用于追溯。</p>'
    old.write_text(s.replace('<h1>',note+'<h1>',1),encoding='utf-8')
