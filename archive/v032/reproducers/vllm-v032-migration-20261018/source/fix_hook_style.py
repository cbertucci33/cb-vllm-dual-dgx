from pathlib import Path
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
p=r/'build/README.md';s=p.read_text();note=s.split('\n\n',1)[0];assert note.startswith('> Migration status:');s=s[len(note)+2:];heading,rest=s.split('\n',1);s=heading+'\n\n'+note+rest;p.write_text(s)
p=r/'tests/parser/engine/test_token_id_scanner.py';s=p.read_text();assert '"partial_control": ["</thi"],' in s;s=s.replace('"partial_control": ["</thi"],','"partial_control": ["</think>"[:5]],');p.write_text(s)
