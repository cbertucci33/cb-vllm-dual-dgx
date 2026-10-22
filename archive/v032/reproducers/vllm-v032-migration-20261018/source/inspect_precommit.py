import sqlite3,json
from pathlib import Path
r=sqlite3.connect('file:/opt/project/.cache/pre-commit/db.db?mode=ro',uri=True)
print('DBTABLES',r.execute("select name from sqlite_master where type='table'").fetchall())
for row in r.execute('select * from repos'):
 print(row)
 p=Path(row[-1]);print('ENVS',[x.name for x in p.iterdir() if x.is_dir() and ('env' in x.name or x.name=='node_modules')])
