from pathlib import Path
import re,subprocess
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
for n in subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=r,text=True).splitlines():
 if not n.startswith('tests/') or any(x in n for x in ['/parser/','/reasoning/','/tool_parsers/','/tool_use/','/entrypoints/','/structured_output/','/kernels/']):continue
 s=(r/n).read_text()
 print('\nFILE',n)
 for i,m in enumerate(re.finditer(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',s,re.M|re.S)):
  print(i+1, 'OURS',m.group(1)[:500].replace('\n',' | '),'THEIRS',m.group(2)[:500].replace('\n',' | '),'LINES',m.group(1).count('\n'),m.group(2).count('\n'))
