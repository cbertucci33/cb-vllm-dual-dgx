from pathlib import Path
import re,sys
side=sys.argv[1]
for name in sys.argv[2:]:
 p=Path(name);s=p.read_text();n=0
 def repl(m):
  global n
  n+=1
  return m.group(1 if side=='ours' else 2)
 s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',repl,s,flags=re.M|re.S)
 p.write_text(s)
 print(name,n,side)
