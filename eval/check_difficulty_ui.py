"""Run real browser difficulty controls with the existing Tauri stub (Chrome required)."""
from pathlib import Path
import tempfile,shutil,subprocess,json
root=Path(__file__).resolve().parents[1] / 'app/ui'
with tempfile.TemporaryDirectory() as td:
 p=Path(td)
 for name in ['main.js','devstub.js','styles.css']:shutil.copy2(root/name,p/name)
 pre='<script>localStorage.setItem("diffs", JSON.stringify(["Easy","Normal","Hard","Expert","ExpertPlus"]));localStorage.setItem("plusTiers","6");</script>'
 checks=r'''<script>setTimeout(() => {
 try {
  const assert=(v,m)=>{if(!v)throw Error(m)};
  assert(chosenDiffs().length===5 && state.plusTiers===0,'saved over-limit migration');
  assert($('plusPlus').disabled,'limit disables plus button');
  state.diffs=new Set(['ExpertPlus']);state.plusTiers=0;renderChips();renderPlus();
  $('plusPlus').click();$('plusPlus').click();
  assert(JSON.stringify(chosenDiffs())===JSON.stringify(['ExpertPlus','ExpertPlus2','ExpertPlus3']),'three tiers coexist');
  assert($('plusLabel').textContent==='Expert++, Expert+++','explicit extra tier labels');
  $('plusPlus').click();$('plusPlus').click();$('plusPlus').click();
  assert(chosenDiffs().length===5 && $('plusPlus').disabled,'cannot choose sixth tier');
  const easy=[...$('diffChips').children].find(b=>b.textContent==='Easy');
  assert(easy.disabled,'unselected chip disabled');easy.onclick();
  assert(chosenDiffs().length===5,'handler enforces cap');
  $('plusMinus').click();
  [...$('diffChips').children].find(b=>b.textContent==='Easy').click();
  assert(chosenDiffs().length===5 && state.diffs.has('Easy'),'freed slot can be selected');
  state.diffs=new Set();state.plusTiers=1;renderChips();renderPlus();$('plusMinus').click();
  assert(chosenDiffs().length===1 && state.diffs.has('ExpertPlus'),'last-tier fallback');
  document.body.dataset.qa='PASS';
 } catch(e){document.body.dataset.qa='FAIL:'+e.message;}
},350);</script>'''
 html=(root/'index.html').read_text().replace('<script src="main.js">',pre+'<script src="devstub.js"></script><script src="main.js">').replace('</body>',checks+'</body>')
 (p/'index.html').write_text(html)
 result=subprocess.run(['google-chrome','--headless=new','--disable-gpu','--no-sandbox','--user-data-dir='+str(p/'profile'),'--virtual-time-budget=1200','--dump-dom',(p/'index.html').as_uri()],capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stderr[-1000:]
 assert 'data-qa="PASS"' in result.stdout,result.stdout[-4000:]
 print('PASS: browser UI migration, +/++/+++ coexistence, five-tier cap, disabled controls, freed slots, nonempty selection')
